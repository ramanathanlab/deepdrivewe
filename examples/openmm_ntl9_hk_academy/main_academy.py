"""Academy-based NTL9 protein folding workflow using OpenMM and Huber-Kim resampling.

This script implements the decentralized multi-agent architecture described in
https://github.com/braceal/deepdrivewe-academy/tree/main/examples/minimal_pattern,
extended with real OpenMM simulations, CVAE training, and weighted ensemble resampling.

Agent Topology
--------------
::

    main()
      ├── register + launch ──> SimulationAgent × N  (one per trajectory)
      ├── register + launch ──> TrainingAgent         (GPU node, CVAE training)
      └── register + launch ──> InferenceAgent        (GPU node, WE resampling)

    SimulationAgent ──SimResult──> TrainingAgent.receive_simulation_data()
    SimulationAgent ──SimResult──> InferenceAgent.receive_simulation_data()
    TrainingAgent   ──TrainResult──> InferenceAgent.receive_model_weights()
    InferenceAgent  ──SimMetadata──> SimulationAgent.simulate()   (next iter)
    main()          ──await manager.wait((inference_handle,))──>  blocks until done

Circular dependencies (SimulationAgent ↔ InferenceAgent) are resolved by
using the register → get_handle → launch pattern from the Academy framework:
mailboxes are created for all agents first, handles are obtained before
instantiation, and agents are launched last with all handles already in hand.

Usage
-----
::

    python examples/openmm_ntl9_hk_academy/main_academy.py -c config_minimal.yaml

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
from deepdrivewe.academy_agents.config import InferenceAgentConfig
from deepdrivewe.academy_agents.config import SimulationPoolConfig
from deepdrivewe.academy_agents.config import TrainingAgentConfig
from deepdrivewe.academy_agents.inference import InferenceAgent
from deepdrivewe.academy_agents.inference import InferenceAgentConfig as _InfCfg
from deepdrivewe.academy_agents.simulation import SimulationAgent
from deepdrivewe.academy_agents.training import TrainingAgent
from deepdrivewe.academy_agents.training import TrainingAgentConfig as _TrnCfg
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
    num_simulations: int = Field(
        default=4,
        description='Number of parallel SimulationAgents to launch.',
    )
    max_retries: int = Field(default=3, description='Max retries for failed sims')
    basis_states: BasisStates
    basis_state_initializer: RMSDBasisStateInitializer
    simulation_config: SimulationConfig
    inference_config: InferenceConfig
    target_states: list[TargetState]
    # Optional override dicts for the new decentralized agents
    training_agent_config: dict | None = Field(
        default=None,
        description='Extra TrainingAgentConfig fields (dict). '
        'None uses defaults.',
    )
    inference_agent_config: dict | None = Field(
        default=None,
        description='Extra InferenceAgentConfig fields (dict). '
        'None uses defaults.',
    )


async def run_workflow(cfg: ExperimentSettings) -> None:
    """Run the decentralized Academy workflow.

    This implements the register → get_handle → launch pattern described
    in the minimal_pattern example, extended with real WE simulation logic.
    """
    logging.info('Starting decentralized Academy NTL9 folding workflow')

    # ------------------------------------------------------------------
    # Setup: output directory, checkpointing, ensemble state
    # ------------------------------------------------------------------
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = EnsembleCheckpointer(output_dir=cfg.output_dir)
    checkpoint = checkpointer.latest_checkpoint()

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

    # ------------------------------------------------------------------
    # WE algorithm components (binner / resampler / recycler)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-agent configuration objects
    # ------------------------------------------------------------------
    sim_pool_config = SimulationPoolConfig(
        num_workers=cfg.num_simulations,
        max_retries=cfg.max_retries,
        retry_delay=1.0,
        output_dir=cfg.output_dir / 'simulations',
        simulation_config=cfg.simulation_config.openmm_config,
        reference_file=cfg.simulation_config.reference_file,
        cutoff_angstrom=cfg.simulation_config.cutoff_angstrom,
        mda_selection=cfg.simulation_config.mda_selection,
        openmm_selection=cfg.simulation_config.openmm_selection,
    )

    # Build TrainingAgentConfig (use simple dataclass from training module)
    raw_train_cfg = cfg.training_agent_config or {}
    train_agent_cfg = _TrnCfg(
        output_dir=cfg.output_dir / 'training',
        **raw_train_cfg,
    )

    # Build InferenceAgentConfig (use simple dataclass from inference module)
    raw_inf_cfg = cfg.inference_agent_config or {}
    inf_agent_cfg = _InfCfg(
        output_dir=cfg.output_dir / 'inference',
        **raw_inf_cfg,
    )

    # ------------------------------------------------------------------
    # Academy manager + decentralized agent launch
    # ------------------------------------------------------------------
    logging.info('Launching Academy agents (decentralized topology)...')

    async with await Manager.from_exchange_factory(
        # Use LocalExchangeFactory for local testing; swap to
        # HybridExchangeFactory(redis_url=...) for HPC deployments.
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:

        # ------------------------------------------------------------------
        # Phase 1: Register all agents (creates mailboxes, no instantiation)
        #
        # This is required to resolve the circular dependency:
        #   SimulationAgent ──> InferenceAgent ──> SimulationAgent
        #
        # Registering creates each agent's mailbox and returns a registration
        # object from which a Handle can be obtained — even before the agent
        # is running. This is the key insight from the minimal_pattern example.
        # ------------------------------------------------------------------
        reg_inference = await manager.register_agent(InferenceAgent)
        reg_training = await manager.register_agent(TrainingAgent)
        reg_simulations = await asyncio.gather(
            *[
                manager.register_agent(SimulationAgent)
                for _ in range(cfg.num_simulations)
            ],
        )

        logging.info(
            f'Registered {len(reg_simulations)} SimulationAgent(s), '
            '1 TrainingAgent, 1 InferenceAgent',
        )

        # ------------------------------------------------------------------
        # Phase 2: Get handles BEFORE launching
        #
        # Handles are mailbox references — they can be passed to agent
        # constructors even before the target agent has been instantiated.
        # ------------------------------------------------------------------
        inference_handle = manager.get_handle(reg_inference)
        training_handle = manager.get_handle(reg_training)
        simulation_handles = [
            manager.get_handle(reg) for reg in reg_simulations
        ]

        # ------------------------------------------------------------------
        # Phase 3: Launch agents with all handles already resolved
        #
        # Launch order:
        #   1. InferenceAgent — owns the iteration loop; loads pretrained
        #      model on startup; must be ready before simulations start.
        #   2. TrainingAgent  — loads CVAE model on startup.
        #   3. SimulationAgents — dispatched in parallel via asyncio.gather.
        # ------------------------------------------------------------------

        # 1. InferenceAgent
        inference_handle = await manager.launch(
            InferenceAgent,
            registration=reg_inference,
            args=(
                cfg.num_simulations,    # num_simulations (batch size)
                cfg.num_iterations,     # max_iterations
                simulation_handles,     # list[Handle[SimulationAgent]]
                inf_agent_cfg,          # InferenceAgentConfig
                binner,                 # Binner
                resampler,              # Resampler
                recycler,               # Recycler
                ensemble,               # WeightedEnsemble (initial state)
                checkpointer,           # EnsembleCheckpointer
            ),
        )
        logging.info('Launched InferenceAgent')

        # 2. TrainingAgent
        training_handle = await manager.launch(
            TrainingAgent,
            registration=reg_training,
            args=(
                inference_handle,       # Handle[InferenceAgent]
                train_agent_cfg,        # TrainingAgentConfig
            ),
        )
        logging.info('Launched TrainingAgent')

        # 3. SimulationAgents (parallel launch)
        simulation_agents = await asyncio.gather(
            *[
                manager.launch(
                    SimulationAgent,
                    registration=reg,
                    args=(
                        sim_pool_config,    # SimulationPoolConfig
                        training_handle,    # Handle[TrainingAgent]
                        inference_handle,   # Handle[InferenceAgent]
                    ),
                )
                for reg in reg_simulations
            ],
        )
        logging.info(f'Launched {len(simulation_agents)} SimulationAgent(s)')

        # ------------------------------------------------------------------
        # Kick off iteration 1
        #
        # The initial SimMetadata objects come from the ensemble's next_sims
        # (either basis states for a fresh run, or the last checkpoint's
        # next_sims when resuming). We dispatch them concurrently.
        # ------------------------------------------------------------------
        initial_sims = ensemble.next_sims

        logging.info(
            f'Dispatching {len(initial_sims)} simulation(s) '
            f'to kick off iteration {ensemble.iteration}...',
        )
        await asyncio.gather(
            *[
                # Round-robin across agents if more sims than agents
                simulation_agents[idx % len(simulation_agents)].simulate(sim)
                for idx, sim in enumerate(initial_sims)
            ],
        )

        # ------------------------------------------------------------------
        # Block until the InferenceAgent signals completion
        #
        # The InferenceAgent's @loop calls shutdown.set() after
        # max_iterations. manager.wait() returns when the agent exits,
        # and the async context manager cascades shutdown to all other agents.
        # ------------------------------------------------------------------
        logging.info(
            'Simulations dispatched. '
            'Waiting for InferenceAgent to signal completion...',
        )
        await manager.wait((inference_handle,))

    logging.info('All agents shut down. Workflow complete.')


def main() -> None:
    """Main entry point."""
    parser = ArgumentParser(
        description='Run NTL9 folding with decentralized Academy agents',
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
    logging.info('Academy NTL9 Folding Workflow (decentralized agent topology)')
    logging.info('=' * 80)
    logging.info(f'Configuration:    {args.config}')
    logging.info(f'Output directory: {cfg.output_dir}')
    logging.info(f'Iterations:       {cfg.num_iterations}')
    logging.info(f'Simulations:      {cfg.num_simulations}')
    logging.info('=' * 80)

    try:
        asyncio.run(run_workflow(cfg))
        logging.info('Workflow completed successfully!')
    except Exception as e:
        logging.error(f'Workflow failed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
