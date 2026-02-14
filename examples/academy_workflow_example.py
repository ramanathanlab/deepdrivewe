"""Example script demonstrating Academy-based weighted ensemble workflow.

This example shows how to set up and run a weighted ensemble simulation
using the Academy framework with the following agent hierarchy:

    OrchestratorAgent
        ├── SimulationPoolAgent
        │   ├── SimulationAgent (worker 1)
        │   ├── SimulationAgent (worker 2)
        │   └── SimulationAgent (worker N)
        └── EnsembleManagerAgent

The workflow demonstrates:
1. Launching agents using Academy's Manager and LocalExchangeFactory
2. Coordinating simulation execution across multiple workers
3. Managing weighted ensemble state and resampling
4. Monitoring workflow progress
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from deepdrivewe import BasisStates
from deepdrivewe import TargetState
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents import AcademyWorkflowConfig
from deepdrivewe.academy_agents import EnsembleManagerAgent
from deepdrivewe.academy_agents import OrchestratorAgent
from deepdrivewe.academy_agents import SimulationAgent
from deepdrivewe.academy_agents import SimulationPoolAgent
from deepdrivewe.academy_agents import SimulationPoolConfig
from deepdrivewe.binners.rectilinear import RectilinearBinner
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.recyclers.low import LowRecycler
from deepdrivewe.resamplers.huber_kim import HuberKimResampler
from deepdrivewe.simulation.openmm import OpenMMConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the Academy-based weighted ensemble workflow."""
    # Configuration
    output_dir = Path('output/academy_example')
    output_dir.mkdir(parents=True, exist_ok=True)

    # OpenMM simulation configuration
    openmm_config = OpenMMConfig(
        simulation_length_ns=0.01,  # 10 ps for testing
        report_interval_ps=1.0,
        platform='CPU',
    )

    # Simulation pool configuration
    sim_pool_config = SimulationPoolConfig(
        num_workers=4,
        max_retries=2,
        retry_delay=1.0,
        output_dir=output_dir / 'simulations',
        simulation_config=openmm_config,
    )

    # Workflow configuration
    workflow_config = AcademyWorkflowConfig(
        output_dir=output_dir,
        num_iterations=5,
        checkpoint_interval=1,
        simulation_pool_config=sim_pool_config,
    )

    # Initialize weighted ensemble components
    # Note: These would normally be loaded from configuration files
    basis_states = BasisStates(
        basis_state_dir=Path('data/basis_states'),
        num_basis_states=1,
    )

    target_state = TargetState(
        target_pcoord=[10.0],  # Example target progress coordinate
    )

    # Create binner, resampler, and recycler
    binner = RectilinearBinner(
        bin_edges=[[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]],
    )

    resampler = HuberKimResampler(
        target_count=4,  # Target 4 simulations per bin
    )

    recycler = LowRecycler(
        target_pcoord=target_state.target_pcoord,
    )

    # Initialize weighted ensemble
    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        target_state=target_state,
        num_iterations=workflow_config.num_iterations,
    )

    # Initialize checkpointer
    checkpointer = EnsembleCheckpointer(
        checkpoint_file=output_dir / 'west.h5',
    )

    logger.info('Starting Academy-based weighted ensemble workflow')

    # Create Academy manager with local exchange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        logger.info('Launched Academy Manager')

        # Launch simulation worker agents
        workers = []
        for i in range(sim_pool_config.num_workers):
            worker = await manager.launch(
                SimulationAgent,
                config=sim_pool_config,
            )
            workers.append(worker)
            logger.info(f'Launched SimulationAgent worker {i}')

        # Launch simulation pool agent
        simulation_pool = await manager.launch(
            SimulationPoolAgent,
            config=sim_pool_config,
            workers=workers,
        )
        logger.info('Launched SimulationPoolAgent')

        # Launch ensemble manager agent
        ensemble_manager = await manager.launch(
            EnsembleManagerAgent,
            ensemble=ensemble,
            binner=binner,
            resampler=resampler,
            recycler=recycler,
        )
        logger.info('Launched EnsembleManagerAgent')

        # Launch orchestrator agent
        orchestrator = await manager.launch(
            OrchestratorAgent,
            config=workflow_config,
            simulation_pool=simulation_pool,
            ensemble_manager=ensemble_manager,
            checkpointer=checkpointer,
        )
        logger.info('Launched OrchestratorAgent')

        # Start the workflow
        await orchestrator.start_workflow()
        logger.info('Workflow started')

        # Run iterations
        for iteration in range(workflow_config.num_iterations):
            logger.info(f'Starting iteration {iteration}')

            # Advance iteration
            success = await orchestrator.advance_iteration()

            if not success:
                logger.info('Workflow complete')
                break

            # Get status
            status = await orchestrator.get_status()
            logger.info(f'Status: {status}')

        # Check completion
        is_complete = await orchestrator.check_completion()
        logger.info(f'Workflow complete: {is_complete}')

        # Get final status
        final_status = await orchestrator.get_status()
        logger.info(f'Final status: {final_status}')

    logger.info('Academy workflow example complete')


if __name__ == '__main__':
    asyncio.run(main())

