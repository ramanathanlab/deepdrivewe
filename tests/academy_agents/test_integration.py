"""Integration tests for Academy agents.

These tests verify that agents can be launched, communicate, and execute
actions correctly using the Academy framework.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from deepdrivewe import BasisStates
from deepdrivewe import TargetState
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents import EnsembleManagerAgent
from deepdrivewe.academy_agents import SimulationAgent
from deepdrivewe.academy_agents import SimulationPoolAgent
from deepdrivewe.academy_agents import SimulationPoolConfig
from deepdrivewe.binners.rectilinear import RectilinearBinner
from deepdrivewe.recyclers.low import LowRecycler
from deepdrivewe.resamplers.huber_kim import HuberKimResampler
from deepdrivewe.simulation.openmm import OpenMMConfig


@pytest.mark.asyncio
async def test_simulation_agent_launch(tmp_path: Path) -> None:
    """Test that SimulationAgent can be launched via Academy Manager."""
    config = SimulationPoolConfig(
        num_workers=1,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=OpenMMConfig(
            simulation_length_ns=0.001,
            report_interval_ps=0.1,
            platform='CPU',
        ),
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch a simulation agent
        agent = await manager.launch(SimulationAgent, config=config)

        # Test that we can call actions
        is_available = await agent.is_available()
        assert is_available is True


@pytest.mark.asyncio
async def test_simulation_pool_agent_launch(tmp_path: Path) -> None:
    """Test that SimulationPoolAgent can be launched with workers."""
    config = SimulationPoolConfig(
        num_workers=2,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=OpenMMConfig(
            simulation_length_ns=0.001,
            report_interval_ps=0.1,
            platform='CPU',
        ),
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch worker agents
        workers = []
        for i in range(config.num_workers):
            worker = await manager.launch(SimulationAgent, config=config)
            workers.append(worker)

        # Launch pool agent
        pool = await manager.launch(
            SimulationPoolAgent,
            config=config,
            workers=workers,
        )

        # Test that we can get available workers
        available = await pool.get_available_workers()
        assert len(available) == 2


@pytest.mark.asyncio
async def test_ensemble_manager_agent_launch(tmp_path: Path) -> None:
    """Test that EnsembleManagerAgent can be launched."""
    # Create minimal ensemble components
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=2,
    )

    target_state = TargetState(pcoord=[10.0])

    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        target_states=[target_state],
    )

    binner = RectilinearBinner(
        bins=[0.0, 5.0, 10.0],
        bin_target_counts=2,
    )
    resampler = HuberKimResampler()
    recycler = LowRecycler(target_pcoord=[10.0])

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch ensemble manager
        agent = await manager.launch(
            EnsembleManagerAgent,
            ensemble=ensemble,
            binner=binner,
            resampler=resampler,
            recycler=recycler,
        )

        # Test that we can get iteration
        iteration = await agent.get_current_iteration()
        assert iteration == 0


@pytest.mark.asyncio
async def test_agent_communication(tmp_path: Path) -> None:
    """Test that agents can communicate via handles."""
    config = SimulationPoolConfig(
        num_workers=1,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=OpenMMConfig(
            simulation_length_ns=0.001,
            report_interval_ps=0.1,
            platform='CPU',
        ),
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch worker
        worker = await manager.launch(SimulationAgent, config=config)

        # Launch pool
        pool = await manager.launch(
            SimulationPoolAgent,
            config=config,
            workers=[worker],
        )

        # Test communication: check worker availability through pool
        available = await pool.get_available_workers()
        assert len(available) == 1
        assert available[0] == 0  # First worker index


@pytest.mark.asyncio
async def test_simulation_pool_task_submission(tmp_path: Path) -> None:
    """Test that tasks can be submitted to the simulation pool."""
    config = SimulationPoolConfig(
        num_workers=1,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=OpenMMConfig(
            simulation_length_ns=0.001,
            report_interval_ps=0.1,
            platform='CPU',
        ),
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch worker
        worker = await manager.launch(SimulationAgent, config=config)

        # Launch pool
        pool = await manager.launch(
            SimulationPoolAgent,
            config=config,
            workers=[worker],
        )

        # Create a mock simulation metadata
        metadata = {
            'sim_id': 'test_sim_001',
            'iteration': 0,
            'walker_id': 0,
            'weight': 1.0,
            'pcoord': [0.0],
            'basis_state_id': 0,
        }

        # Submit a task
        sim_id = await pool.submit_simulation(metadata)
        assert sim_id == 'test_sim_001'

        # Give it a moment to process
        await asyncio.sleep(0.5)

        # Check that results are available (or still pending)
        all_results = await pool.get_all_results()
        assert isinstance(all_results, dict)


@pytest.mark.asyncio
async def test_ensemble_manager_actions(tmp_path: Path) -> None:
    """Test EnsembleManagerAgent actions."""
    # Create minimal ensemble
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=2,
    )

    target_state = TargetState(pcoord=[10.0])

    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        target_states=[target_state],
    )

    binner = RectilinearBinner(
        bins=[0.0, 5.0, 10.0],
        bin_target_counts=2,
    )
    resampler = HuberKimResampler()
    recycler = LowRecycler(target_pcoord=[10.0])

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        agent = await manager.launch(
            EnsembleManagerAgent,
            ensemble=ensemble,
            binner=binner,
            resampler=resampler,
            recycler=recycler,
        )

        # Test get_current_iteration
        iteration = await agent.get_current_iteration()
        assert iteration == 0

        # Test get_ensemble_state
        state = await agent.get_ensemble_state()
        assert isinstance(state, dict)
        assert 'iteration' in state
        assert 'num_simulations' in state


