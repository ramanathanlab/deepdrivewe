"""Simple integration tests for Academy agents without requiring full MD setup.

These tests verify basic agent functionality using mocks and minimal setup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from deepdrivewe import BasisStates
from deepdrivewe import TargetState
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents import EnsembleManagerAgent
from deepdrivewe.academy_agents import SimulationPoolConfig
from deepdrivewe.binners.rectilinear import RectilinearBinner
from deepdrivewe.recyclers.low import LowRecycler
from deepdrivewe.resamplers.huber_kim import HuberKimResampler
from deepdrivewe.simulation.openmm import OpenMMConfig


def test_simulation_pool_config_creation(tmp_path: Path) -> None:
    """Test that SimulationPoolConfig can be created and validated."""
    openmm_config = OpenMMConfig(
        simulation_length_ns=0.001,
        report_interval_ps=0.1,
        hardware_platform='CPU',
    )

    config = SimulationPoolConfig(
        num_workers=4,
        max_retries=2,
        retry_delay=1.0,
        output_dir=tmp_path / 'simulations',
        simulation_config=openmm_config,
    )

    assert config.num_workers == 4
    assert config.max_retries == 2
    assert config.retry_delay == 1.0
    # Note: output_dir is created by model_validator in config.py


def test_ensemble_manager_instantiation(tmp_path: Path) -> None:
    """Test that EnsembleManagerAgent can be instantiated with proper components."""
    # Create basis states
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=4,
    )

    # Create target state
    target_state = TargetState(pcoord=[10.0])

    # Create ensemble
    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        target_states=[target_state],
    )

    # Create binner, resampler, recycler
    binner = RectilinearBinner(
        bins=[0.0, 5.0, 10.0],
        bin_target_counts=2,
    )
    resampler = HuberKimResampler()
    recycler = LowRecycler(
        basis_states=basis_states,
        target_threshold=1.0,
    )

    # Create agent
    agent = EnsembleManagerAgent(
        ensemble=ensemble,
        binner=binner,
        resampler=resampler,
        recycler=recycler,
    )

    assert agent is not None
    assert agent.ensemble == ensemble
    assert agent.binner == binner
    assert agent.resampler == resampler
    assert agent.recycler == recycler


def test_weighted_ensemble_initialization(tmp_path: Path) -> None:
    """Test that WeightedEnsemble can be initialized properly."""
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=4,
    )

    target_state = TargetState(pcoord=[10.0])

    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        target_states=[target_state],
    )

    assert ensemble.basis_states == basis_states
    assert len(ensemble.target_states) == 1
    assert ensemble.target_states[0] == target_state
    assert len(ensemble.cur_sims) == 0
    assert len(ensemble.next_sims) == 0


def test_binner_creation() -> None:
    """Test that RectilinearBinner can be created."""
    binner = RectilinearBinner(
        bins=[0.0, 2.5, 5.0, 7.5, 10.0],
        bin_target_counts=4,
    )

    assert binner is not None


def test_resampler_creation() -> None:
    """Test that HuberKimResampler can be created."""
    resampler = HuberKimResampler()
    assert resampler is not None


def test_recycler_creation(tmp_path: Path) -> None:
    """Test that LowRecycler can be created."""
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=4,
    )
    recycler = LowRecycler(
        basis_states=basis_states,
        target_threshold=1.0,
    )
    assert recycler is not None


def test_openmm_config_creation() -> None:
    """Test that OpenMMConfig can be created with various platforms."""
    # CPU platform
    config_cpu = OpenMMConfig(
        simulation_length_ns=0.01,
        report_interval_ps=1.0,
        hardware_platform='CPU',
    )
    assert config_cpu.hardware_platform == 'CPU'

    # CUDA platform (default)
    config_cuda = OpenMMConfig(
        simulation_length_ns=0.01,
        report_interval_ps=1.0,
    )
    assert config_cuda.hardware_platform == 'CUDA'


def test_basis_states_validation(tmp_path: Path) -> None:
    """Test that BasisStates validates initial_ensemble_members."""
    # Valid configuration
    basis_states = BasisStates(
        basis_state_dir=tmp_path / 'basis_states',
        initial_ensemble_members=4,
    )
    assert basis_states.initial_ensemble_members == 4

    # Test that initial_ensemble_members must be >= 1
    with pytest.raises(Exception):  # Pydantic validation error
        BasisStates(
            basis_state_dir=tmp_path / 'basis_states',
            initial_ensemble_members=0,  # Invalid
        )

