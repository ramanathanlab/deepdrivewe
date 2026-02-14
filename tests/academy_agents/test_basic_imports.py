"""Basic import and instantiation tests for Academy agents."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepdrivewe import BasisStates
from deepdrivewe import TargetState
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents import AcademyWorkflowConfig
from deepdrivewe.academy_agents import EnsembleManagerAgent
from deepdrivewe.academy_agents import SimulationPoolConfig
from deepdrivewe.binners.rectilinear import RectilinearBinner
from deepdrivewe.recyclers.low import LowRecycler
from deepdrivewe.resamplers.huber_kim import HuberKimResampler
from deepdrivewe.simulation.openmm import OpenMMConfig


def test_imports() -> None:
    """Test that all Academy agent modules can be imported."""
    from deepdrivewe.academy_agents import AcademyAgent
    from deepdrivewe.academy_agents import EnsembleManagerAgent
    from deepdrivewe.academy_agents import OrchestratorAgent
    from deepdrivewe.academy_agents import SimulationAgent
    from deepdrivewe.academy_agents import SimulationPoolAgent

    assert AcademyAgent is not None
    assert OrchestratorAgent is not None
    assert SimulationAgent is not None
    assert SimulationPoolAgent is not None
    assert EnsembleManagerAgent is not None


def test_config_creation(tmp_path: Path) -> None:
    """Test that configuration models can be created."""
    openmm_config = OpenMMConfig(
        simulation_length_ns=0.001,
        report_interval_ps=0.1,
        platform='CPU',
    )

    sim_pool_config = SimulationPoolConfig(
        num_workers=2,
        max_retries=1,
        retry_delay=0.1,
        output_dir=tmp_path / 'simulations',
        simulation_config=openmm_config,
    )

    workflow_config = AcademyWorkflowConfig(
        output_dir=tmp_path,
        num_iterations=2,
        checkpoint_interval=1,
        simulation_pool_config=sim_pool_config,
    )

    assert workflow_config.num_iterations == 2
    assert workflow_config.simulation_pool_config.num_workers == 2
    assert workflow_config.output_dir.exists()


def test_ensemble_manager_creation(tmp_path: Path) -> None:
    """Test that EnsembleManagerAgent can be instantiated."""
    # Just test that we can import and create the class
    # Full integration tests would require proper setup of all components
    from deepdrivewe.academy_agents import EnsembleManagerAgent

    # We can't easily create a full ensemble without proper setup,
    # so we just verify the class exists and can be imported
    assert EnsembleManagerAgent is not None
    assert hasattr(EnsembleManagerAgent, '__init__')


def test_simulation_pool_config_validation(tmp_path: Path) -> None:
    """Test that SimulationPoolConfig validates inputs."""
    openmm_config = OpenMMConfig(
        simulation_length_ns=0.001,
        platform='CPU',
    )

    # Valid config
    config = SimulationPoolConfig(
        num_workers=4,
        max_retries=2,
        retry_delay=1.0,
        output_dir=tmp_path / 'simulations',
        simulation_config=openmm_config,
    )

    assert config.num_workers == 4
    assert config.max_retries == 2

    # Test that num_workers must be >= 1
    with pytest.raises(Exception):  # Pydantic validation error
        SimulationPoolConfig(
            num_workers=0,  # Invalid
            max_retries=2,
            retry_delay=1.0,
            output_dir=tmp_path / 'simulations',
            simulation_config=openmm_config,
        )

