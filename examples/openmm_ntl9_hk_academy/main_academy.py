"""Academy-based NTL9 protein folding workflow using OpenMM and Huber-Kim resampling.

This script implements the Academy agents architecture with the following hierarchy:

Agent Topology
--------------
::

    OrchestratorAgent (Workflow Coordinator)
    ├── SimulationPoolAgent (Task Distribution)
    │   ├── SimulationAgent (Worker 1)
    │   ├── SimulationAgent (Worker 2)
    │   └── SimulationAgent (Worker N)
    ├── EnsembleManagerAgent (WE State Management)
    └── AnalysisPoolAgent (Analysis Coordination) [Optional]
        ├── CVAEAnalyzer (Latent Space Projection)
        └── LOFAnalyzer (Anomaly Detection)

Workflow
--------
1. Initialization: Load/create weighted ensemble, launch agents
2. Iteration Loop:
   - Submit simulations to pool
   - Execute simulations in parallel
   - Run analysis on results (CVAE → LOF) [if enabled]
   - Apply resampling (Huber-Kim)
   - Update ensemble state
   - Checkpoint results
3. Shutdown: Graceful agent termination

Usage
-----
::

    python examples/openmm_ntl9_hk_academy/main_academy.py --config config_minimal.yaml

"""

from __future__ import annotations

import asyncio
import logging
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager
from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe import BasisStates
from deepdrivewe import EnsembleCheckpointer
from deepdrivewe import TargetState
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents import AcademyWorkflowConfig
from deepdrivewe.academy_agents import AnalysisPoolAgent
from deepdrivewe.academy_agents import AnalysisPoolConfig
from deepdrivewe.academy_agents import EnsembleManagerAgent
from deepdrivewe.academy_agents import OrchestratorAgent
from deepdrivewe.academy_agents import SimulationAgent
from deepdrivewe.academy_agents import SimulationPoolAgent
from deepdrivewe.academy_agents import SimulationPoolConfig
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.examples.openmm_ntl9_hk.inference import InferenceConfig
from deepdrivewe.examples.openmm_ntl9_hk.main import RMSDBasisStateInitializer
from deepdrivewe.examples.openmm_ntl9_hk.simulate import SimulationConfig
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import HuberKimResampler


class ExperimentSettings(BaseModel):
    """Settings for the NTL9 folding experiment."""

    output_dir: Path = Field(description='Output directory for results')
    num_iterations: int = Field(description='Number of WE iterations to run')
    max_retries: int = Field(default=3, description='Max retries for failed sims')
    basis_states: BasisStates
    basis_state_initializer: RMSDBasisStateInitializer
    simulation_config: SimulationConfig
    inference_config: InferenceConfig
    target_states: list[TargetState]
    academy_config: AcademyConfig | None = Field(
        default=None,
        description='Academy agents configuration',
    )
    analysis_config: AnalysisConfig | None = Field(
        default=None,
        description='Analysis pool configuration (optional)',
    )


class AcademyConfig(BaseModel):
    """Configuration for Academy agents."""

    num_workers: int = Field(default=2, description='Number of simulation workers')
    exchange_type: str = Field(default='local', description='Exchange type (local or redis)')


class AnalysisConfig(BaseModel):
    """Configuration for analysis pool."""

    enabled_analyzers: list[str] = Field(
        default_factory=list,
        description='List of enabled analyzers (cvae, lof)',
    )
    analyzer_configs: dict = Field(
        default_factory=dict,
        description='Configuration for each analyzer',
    )


async def run_academy_workflow(cfg: ExperimentSettings) -> None:
    """Run the Academy-based weighted ensemble workflow."""
    logging.info('Starting Academy-based NTL9 folding workflow')

    # Setup output directory and checkpointing
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = EnsembleCheckpointer(output_dir=cfg.output_dir)
    checkpoint = checkpointer.latest_checkpoint()

    # Initialize or load ensemble
    if checkpoint is None:
        ensemble = WeightedEnsemble(
            basis_states=cfg.basis_states,
            target_states=cfg.target_states,
        )
        ensemble.initialize_basis_states(cfg.basis_state_initializer)
        logging.info('Initialized new weighted ensemble')
    else:
        ensemble = checkpointer.load(checkpoint)
        logging.info(f'Loaded ensemble from checkpoint: {checkpoint}')

    logging.info(f'Initial ensemble size: {len(ensemble.next_sims)}')

    # Initialize WE components
    binner = RectilinearBinner(
        bins=[0.0, 1.00]
        + [1.10 + 0.1 * i for i in range(35)]
        + [4.60 + 0.2 * i for i in range(10)]
        + [6.60 + 0.6 * i for i in range(6)]
        + [float('inf')],
        bin_target_counts=cfg.inference_config.sims_per_bin,
    )

    resampler = HuberKimResampler(
        sims_per_bin=cfg.inference_config.sims_per_bin,
        max_allowed_weight=cfg.inference_config.max_allowed_weight,
        min_allowed_weight=cfg.inference_config.min_allowed_weight,
    )

    recycler = LowRecycler(
        basis_states=ensemble.basis_states,
        target_threshold=cfg.target_states[0].pcoord[0],
    )

    # Configure Academy agents
    academy_cfg = cfg.academy_config or AcademyConfig()

    sim_pool_config = SimulationPoolConfig(
        num_workers=academy_cfg.num_workers,
        max_retries=cfg.max_retries,
        retry_delay=1.0,
        output_dir=cfg.output_dir / 'simulations',
        simulation_config=cfg.simulation_config.openmm_config,
        reference_file=cfg.simulation_config.reference_file,
        cutoff_angstrom=cfg.simulation_config.cutoff_angstrom,
        mda_selection=cfg.simulation_config.mda_selection,
        openmm_selection=cfg.simulation_config.openmm_selection,
    )

    workflow_config = AcademyWorkflowConfig(
        output_dir=cfg.output_dir,
        num_iterations=cfg.num_iterations,
        checkpoint_interval=1,
        simulation_pool_config=sim_pool_config,
    )

    # Create Academy manager with local exchange
    logging.info('Launching Academy agents...')

    # We need enough workers for all agents:
    # - num_workers SimulationAgent workers
    # - 1 SimulationPoolAgent
    # - 1 EnsembleManagerAgent
    # - 1 AnalysisPoolAgent (if enabled)
    # - 1 OrchestratorAgent
    # Total: num_workers + 4
    num_executor_workers = academy_cfg.num_workers + 4

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=num_executor_workers),
    ) as manager:
        logging.info('Launched Academy Manager')

        # Launch simulation worker agents
        workers = []
        for i in range(academy_cfg.num_workers):
            worker = await manager.launch(
                SimulationAgent,
                kwargs={'config': sim_pool_config},
            )
            workers.append(worker)
            logging.info(f'Launched SimulationAgent worker {i}')

        # Launch simulation pool agent
        simulation_pool = await manager.launch(
            SimulationPoolAgent,
            kwargs={'config': sim_pool_config, 'workers': workers},
        )
        logging.info('Launched SimulationPoolAgent')

        # Launch ensemble manager agent
        ensemble_manager = await manager.launch(
            EnsembleManagerAgent,
            kwargs={
                'ensemble': ensemble,
                'binner': binner,
                'resampler': resampler,
                'recycler': recycler,
            },
        )
        logging.info('Launched EnsembleManagerAgent')

        # Launch analysis pool agent (if enabled)
        analysis_agent = None
        if cfg.analysis_config is not None:
            analysis_pool_config = AnalysisPoolConfig(
                output_dir=cfg.output_dir / 'analysis',
                enabled_analyzers=cfg.analysis_config.enabled_analyzers,
                analyzer_configs=cfg.analysis_config.analyzer_configs,
            )
            analysis_agent = await manager.launch(
                AnalysisPoolAgent,
                kwargs={
                    'output_dir': analysis_pool_config.output_dir,
                    'enabled_analyzers': analysis_pool_config.enabled_analyzers,
                    'analyzer_configs': analysis_pool_config.analyzer_configs,
                },
            )
            logging.info(
                f'Launched AnalysisPoolAgent with analyzers: '
                f'{analysis_pool_config.enabled_analyzers}',
            )

        # Launch orchestrator agent
        orchestrator = await manager.launch(
            OrchestratorAgent,
            kwargs={
                'config': workflow_config,
                'simulation_pool': simulation_pool,
                'ensemble_manager': ensemble_manager,
                'checkpointer': checkpointer,
                'analysis_pool': analysis_agent,
            },
        )
        logging.info('Launched OrchestratorAgent')

        # Start the workflow
        await orchestrator.start_workflow()
        logging.info('Workflow started')

        # Run iterations
        for iteration in range(cfg.num_iterations):
            logging.info(f'Starting iteration {iteration + 1}/{cfg.num_iterations}')

            # Advance iteration
            success = await orchestrator.advance_iteration()

            if not success:
                logging.info('Workflow complete (no more simulations)')
                break

            # Get status
            status = await orchestrator.get_status()
            logging.info(
                f'Iteration {iteration + 1} complete. '
                f'Current iteration: {status["current_iteration"]}, '
                f'Complete: {status["workflow_complete"]}',
            )

        # Check completion
        is_complete = await orchestrator.check_completion()
        logging.info(f'Workflow complete: {is_complete}')

        # Get final status
        final_status = await orchestrator.get_status()
        logging.info(f'Final status: {final_status}')

    logging.info('All agents shut down. Workflow complete.')


def main() -> None:
    """Main entry point."""
    parser = ArgumentParser(
        description='Run NTL9 folding with Academy agents',
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Path to configuration YAML file',
    )
    args = parser.parse_args()

    cfg = ExperimentSettings.from_yaml(args.config)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(cfg.output_dir / 'runtime.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info('=' * 80)
    logging.info('Academy NTL9 Folding Workflow (Academy agents architecture)')
    logging.info('=' * 80)
    logging.info(f'Configuration:    {args.config}')
    logging.info(f'Output directory: {cfg.output_dir}')
    logging.info(f'Iterations:       {cfg.num_iterations}')
    academy_cfg = cfg.academy_config or AcademyConfig()
    logging.info(f'Workers:          {academy_cfg.num_workers}')
    if cfg.analysis_config:
        logging.info(f'Analysis:         {cfg.analysis_config.enabled_analyzers}')
    logging.info('=' * 80)

    try:
        asyncio.run(run_academy_workflow(cfg))
        logging.info('Workflow completed successfully!')
    except Exception as e:
        logging.error(f'Workflow failed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
