"""Academy-based NTL9 protein folding example using OpenMM and Huber-Kim resampling.

This script demonstrates the complete Academy agents workflow for weighted ensemble
simulations, replacing the Colmena-based implementation with Academy agents.
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
from deepdrivewe.academy_agents.config import AcademyWorkflowConfig
from deepdrivewe.academy_agents.config import SimulationPoolConfig
from deepdrivewe.academy_agents.ensemble import EnsembleManagerAgent
from deepdrivewe.academy_agents.orchestrator import OrchestratorAgent
from deepdrivewe.academy_agents.simulation import SimulationAgent
from deepdrivewe.academy_agents.simulation import SimulationPoolAgent
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
    academy_config: dict = Field(
        default_factory=lambda: {'num_workers': 2, 'exchange_type': 'local'},
    )


async def run_academy_workflow(cfg: ExperimentSettings) -> None:
    """Run the Academy-based weighted ensemble workflow."""
    logging.info('Starting Academy-based NTL9 folding workflow')
    
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the checkpoint manager
    checkpointer = EnsembleCheckpointer(output_dir=cfg.output_dir)
    
    # Check if a checkpoint exists
    checkpoint = checkpointer.latest_checkpoint()
    
    if checkpoint is None:
        # Initialize the weighted ensemble
        ensemble = WeightedEnsemble(
            basis_states=cfg.basis_states,
            target_states=cfg.target_states,
        )
        
        # Initialize the simulations with the basis states
        ensemble.initialize_basis_states(cfg.basis_state_initializer)
        logging.info('Initialized new weighted ensemble')
    else:
        # Load the ensemble from a checkpoint if it exists
        ensemble = checkpointer.load(checkpoint)
        logging.info(f'Loaded ensemble from checkpoint {checkpoint}')
    
    # Print the input states
    logging.info(f'Basis states: {ensemble.basis_states}')
    logging.info(f'Target states: {ensemble.target_states}')
    logging.info(f'Initial ensemble size: {len(ensemble.next_sims)}')
    
    # Create binner, resampler, and recycler
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
    
    # Create simulation pool configuration
    sim_pool_config = SimulationPoolConfig(
        num_workers=cfg.academy_config['num_workers'],
        max_retries=cfg.max_retries,
        retry_delay=1.0,
        output_dir=cfg.output_dir / 'simulations',
        simulation_config=cfg.simulation_config.openmm_config,
        reference_file=cfg.simulation_config.reference_file,
        cutoff_angstrom=cfg.simulation_config.cutoff_angstrom,
        mda_selection=cfg.simulation_config.mda_selection,
        openmm_selection=cfg.simulation_config.openmm_selection,
    )

    # Create Academy workflow configuration
    workflow_config = AcademyWorkflowConfig(
        num_iterations=cfg.num_iterations,
        checkpoint_interval=1,
        output_dir=cfg.output_dir,
        simulation_pool_config=sim_pool_config,
    )
    
    logging.info('Launching Academy agents...')
    
    # Launch Academy agents
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        # Launch simulation worker agents
        workers = []
        for i in range(sim_pool_config.num_workers):
            worker = await manager.launch(SimulationAgent, args=(sim_pool_config,))
            workers.append(worker)
            logging.info(f'Launched SimulationAgent worker {i}')
        
        # Launch simulation pool agent
        pool_agent = await manager.launch(
            SimulationPoolAgent,
            args=(sim_pool_config, workers),
        )
        logging.info('Launched SimulationPoolAgent')

        # Launch ensemble manager agent
        ensemble_agent = await manager.launch(
            EnsembleManagerAgent,
            args=(ensemble, binner, resampler, recycler),
        )
        logging.info('Launched EnsembleManagerAgent')

        # Launch orchestrator agent (pass handles, not agents)
        orchestrator = await manager.launch(
            OrchestratorAgent,
            args=(workflow_config, pool_agent, ensemble_agent, checkpointer),
        )
        logging.info('Launched OrchestratorAgent')

        # Start the workflow
        logging.info('Starting weighted ensemble workflow...')
        await orchestrator.start_workflow()

        # Run iterations
        logging.info('Running weighted ensemble iterations...')
        for iteration in range(cfg.num_iterations):
            logging.info(f'Starting iteration {iteration + 1}/{cfg.num_iterations}')

            # Advance iteration
            success = await orchestrator.advance_iteration()

            if not success:
                logging.info('Workflow completed early')
                break

            # Get status
            status = await orchestrator.get_status()
            logging.info(
                f"Iteration {status['current_iteration']}/{status['total_iterations']} - "
                f"Ensemble: {status['ensemble_state']['num_current_sims']} current sims, "
                f"{status['ensemble_state']['num_next_sims']} next sims"
            )

        # Get final status
        final_status = await orchestrator.get_status()
        logging.info(f'Workflow completed!')
        logging.info(f'Final status: {final_status}')

        # Shutdown agents
        logging.info('Shutting down agents...')
        await manager.shutdown(orchestrator, blocking=True)
        await manager.shutdown(ensemble_agent, blocking=True)
        await manager.shutdown(pool_agent, blocking=True)
        for worker in workers:
            await manager.shutdown(worker, blocking=True)

        logging.info('All agents shut down successfully')

    logging.info('Academy workflow completed!')


def main() -> None:
    """Main entry point."""
    parser = ArgumentParser(
        description='Run NTL9 folding with Academy agents'
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Path to configuration YAML file',
    )
    args = parser.parse_args()

    # Load configuration
    cfg = ExperimentSettings.from_yaml(args.config)

    # Save configuration to output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(cfg.output_dir / 'runtime.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info('='*80)
    logging.info('Academy-based NTL9 Protein Folding Workflow')
    logging.info('='*80)
    logging.info(f'Configuration: {args.config}')
    logging.info(f'Output directory: {cfg.output_dir}')
    logging.info(f'Number of iterations: {cfg.num_iterations}')
    logging.info(f'Number of workers: {cfg.academy_config["num_workers"]}')
    logging.info('='*80)

    # Run the async workflow
    try:
        asyncio.run(run_academy_workflow(cfg))
        logging.info('Workflow completed successfully!')
    except Exception as e:
        logging.error(f'Workflow failed with error: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()


