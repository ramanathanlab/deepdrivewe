"""Configuration models for Academy-based workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe.simulation.openmm import OpenMMConfig


class SimulationPoolConfig(BaseModel):
    """Configuration for the simulation pool agent.

    Parameters
    ----------
    num_workers : int
        Number of simulation worker agents to spawn.
    max_retries : int
        Maximum number of retries for failed simulations.
    retry_delay : float
        Delay in seconds between retries.
    output_dir : Path
        Directory to store simulation outputs.
    simulation_config : OpenMMConfig
        Configuration for OpenMM simulations.
    """

    num_workers: int = Field(
        default=4,
        ge=1,
        description='Number of simulation worker agents to spawn.',
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description='Maximum number of retries for failed simulations.',
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description='Delay in seconds between retries.',
    )
    output_dir: Path = Field(
        description='Directory to store simulation outputs.',
    )
    simulation_config: OpenMMConfig = Field(
        description='Configuration for OpenMM simulations.',
    )


class AnalysisPoolConfig(BaseModel):
    """Configuration for the analysis pool agent.

    This configuration will be used in Phase 3 to support CVAE, ANCA,
    and LOF analysis plugins.

    Parameters
    ----------
    output_dir : Path
        Directory to store analysis outputs.
    enabled_analyzers : list[str]
        List of enabled analyzer names (e.g., ['cvae', 'anca', 'lof']).
    analyzer_configs : dict[str, Any]
        Configuration for each analyzer, keyed by analyzer name.
    """

    output_dir: Path = Field(
        description='Directory to store analysis outputs.',
    )
    enabled_analyzers: list[str] = Field(
        default_factory=list,
        description="List of enabled analyzer names (e.g., ['cvae', 'anca', 'lof']).",
    )
    analyzer_configs: dict[str, Any] = Field(
        default_factory=dict,
        description='Configuration for each analyzer, keyed by analyzer name.',
    )


class AcademyWorkflowConfig(BaseModel):
    """Configuration for Academy-based weighted ensemble workflow.

    Parameters
    ----------
    output_dir : Path
        Root directory for all workflow outputs.
    num_iterations : int
        Number of weighted ensemble iterations to run.
    checkpoint_interval : int
        Save ensemble checkpoint every N iterations.
    simulation_pool_config : SimulationPoolConfig
        Configuration for the simulation pool.
    analysis_pool_config : AnalysisPoolConfig
        Configuration for the analysis pool (Phase 3).
    """

    output_dir: Path = Field(
        description='Root directory for all workflow outputs.',
    )
    num_iterations: int = Field(
        ge=1,
        description='Number of weighted ensemble iterations to run.',
    )
    checkpoint_interval: int = Field(
        default=1,
        ge=1,
        description='Save ensemble checkpoint every N iterations.',
    )
    simulation_pool_config: SimulationPoolConfig = Field(
        description='Configuration for the simulation pool.',
    )
    analysis_pool_config: AnalysisPoolConfig | None = Field(
        default=None,
        description='Configuration for the analysis pool (Phase 3).',
    )

    def model_post_init(self, __context: Any) -> None:
        """Create output directories after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation_pool_config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        if self.analysis_pool_config is not None:
            self.analysis_pool_config.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

