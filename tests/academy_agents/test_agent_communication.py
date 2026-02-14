"""Unit tests for Academy agent communication patterns."""

from __future__ import annotations

import asyncio
from pathlib import Path

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


@pytest.fixture
def sim_pool_config(tmp_path: Path) -> SimulationPoolConfig:
    """Create a simulation pool configuration for testing."""
    openmm_config = OpenMMConfig(
        simulation_length_ns=0.001,  # Very short for testing
        report_interval_ps=0.1,
        platform='CPU',
    )

    return SimulationPoolConfig(
        num_workers=2,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=openmm_config,
    )


@pytest.fixture
def weighted_ensemble(tmp_path: Path) -> WeightedEnsemble:
    """Create a weighted ensemble for testing."""
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        num_basis_states=1,
    )

    target_state = TargetState(
        target_pcoord=[10.0],
    )

    return WeightedEnsemble(
        basis_states=basis_states,
        target_state=target_state,
        num_iterations=2,
    )


@pytest.mark.asyncio
async def test_simulation_agent_availability(
    sim_pool_config: SimulationPoolConfig,
) -> None:
    """Test that SimulationAgent reports availability correctly."""
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch simulation agent
        agent = await manager.launch(SimulationAgent, config=sim_pool_config)

        # Check initial availability
        is_available = await agent.is_available()
        assert is_available is True


@pytest.mark.asyncio
async def test_ensemble_manager_get_simulations(
    weighted_ensemble: WeightedEnsemble,
) -> None:
    """Test that EnsembleManagerAgent can return simulations."""
    binner = RectilinearBinner(bin_edges=[[0.0, 5.0, 10.0]])
    resampler = HuberKimResampler(target_count=2)
    recycler = LowRecycler(target_pcoord=[10.0])

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch ensemble manager
        agent = await manager.launch(
            EnsembleManagerAgent,
            ensemble=weighted_ensemble,
            binner=binner,
            resampler=resampler,
            recycler=recycler,
        )

        # Get next simulations
        next_sims = await agent.get_next_simulations()

        # Should return a list of simulation metadata dictionaries
        assert isinstance(next_sims, list)
        assert len(next_sims) > 0
        assert all(isinstance(sim, dict) for sim in next_sims)


@pytest.mark.asyncio
async def test_ensemble_manager_get_iteration(
    weighted_ensemble: WeightedEnsemble,
) -> None:
    """Test that EnsembleManagerAgent returns current iteration."""
    binner = RectilinearBinner(bin_edges=[[0.0, 5.0, 10.0]])
    resampler = HuberKimResampler(target_count=2)
    recycler = LowRecycler(target_pcoord=[10.0])

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch ensemble manager
        agent = await manager.launch(
            EnsembleManagerAgent,
            ensemble=weighted_ensemble,
            binner=binner,
            resampler=resampler,
            recycler=recycler,
        )

        # Get current iteration
        iteration = await agent.get_current_iteration()

        # Should return an integer
        assert isinstance(iteration, int)
        assert iteration >= 0


@pytest.mark.asyncio
async def test_simulation_pool_submit(
    sim_pool_config: SimulationPoolConfig,
) -> None:
    """Test that SimulationPoolAgent can accept simulation submissions."""
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch workers
        workers = []
        for _ in range(2):
            worker = await manager.launch(
                SimulationAgent,
                config=sim_pool_config,
            )
            workers.append(worker)

        # Launch pool
        pool = await manager.launch(
            SimulationPoolAgent,
            config=sim_pool_config,
            workers=workers,
        )

        # Submit a simulation
        metadata = {
            'simulation_id': 'test_sim_001',
            'weight': 1.0,
            'pcoord': [0.0],
        }

        sim_id = await pool.submit_simulation(metadata)

        # Should return the simulation ID
        assert sim_id == 'test_sim_001'

