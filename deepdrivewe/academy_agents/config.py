"""Configuration models for Academy-based workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Sequence

from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe.simulation.openmm import OpenMMConfig


class TrainingAgentConfig(BaseModel):
    """Configuration for the TrainingAgent.

    The TrainingAgent runs on a GPU node and trains the CVAE model
    on simulation data as it arrives (streaming / online training).
    The model stays warm in GPU memory across iterations.

    Parameters
    ----------
    output_dir : Path
        Directory to store CVAE model checkpoints and training logs.
    pretrained_model_path : Path | None
        Path to a pretrained CVAE checkpoint to load on startup.
        If None, the model is initialized from scratch.
    train_frequency : int
        Number of SimResult objects to accumulate before triggering
        a training step. Default is 1 (train on every result).
    cvae_config : dict[str, Any] | None
        Dictionary of ``ConvolutionalVAEConfig`` fields. Passed through
        to the CVAE constructor. If None, default CVAE settings are used.
    """

    output_dir: Path = Field(
        description='Directory to store CVAE model checkpoints and logs.',
    )
    pretrained_model_path: Path | None = Field(
        default=None,
        description='Path to a pretrained CVAE checkpoint to load on startup.',
    )
    train_frequency: int = Field(
        default=1,
        ge=1,
        description='Number of SimResults to accumulate before training.',
    )
    cvae_config: dict[str, Any] | None = Field(
        default=None,
        description='ConvolutionalVAEConfig fields (dict). '
        'None uses CVAE defaults.',
    )

    def model_post_init(self, __context: Any) -> None:
        """Create output directory after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class InferenceAgentConfig(BaseModel):
    """Configuration for the InferenceAgent.

    The InferenceAgent runs on a GPU node and drives the weighted ensemble
    iteration loop. It collects simulation results, runs CVAE inference
    (latent projection), applies WE resampling, saves checkpoints, and
    dispatches the next iteration of simulations.

    A pretrained model should be provided so that the inference agent is
    ready from iteration 1 without waiting for the training agent to
    complete its first training step.

    Parameters
    ----------
    output_dir : Path
        Directory to store inference outputs.
    pretrained_model_path : Path | None
        Path to a pretrained CVAE checkpoint to load on startup.
        Strongly recommended so that inference is available at iteration 1.
    cvae_config : dict[str, Any] | None
        Dictionary of ``ConvolutionalVAEConfig`` fields used during the
        inference (predict) step. If None, default CVAE settings are used.
    """

    output_dir: Path = Field(
        description='Directory to store inference outputs.',
    )
    pretrained_model_path: Path | None = Field(
        default=None,
        description='Path to a pretrained CVAE checkpoint to load on startup. '
        'Strongly recommended for warm startup.',
    )
    cvae_config: dict[str, Any] | None = Field(
        default=None,
        description='ConvolutionalVAEConfig fields (dict). '
        'None uses CVAE defaults.',
    )

    def model_post_init(self, __context: Any) -> None:
        """Create output directory after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


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
    reference_file : Path
        Reference PDB file for RMSD calculation.
    cutoff_angstrom : float
        Cutoff distance for contact map calculation.
    mda_selection : str
        MDAnalysis selection string for atoms.
    openmm_selection : Sequence[str]
        OpenMM selection strings for atoms.
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
    reference_file: Path | None = Field(
        default=None,
        description='Reference PDB file for RMSD calculation (optional).',
    )
    cutoff_angstrom: float = Field(
        default=8.0,
        description='Cutoff distance for contact map calculation.',
    )
    mda_selection: str = Field(
        default='protein and name CA',
        description='MDAnalysis selection string for atoms.',
    )
    openmm_selection: Sequence[str] = Field(
        default=('CA',),
        description='OpenMM selection strings for atoms.',
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
    num_simulations : int
        Number of parallel SimulationAgents (one per trajectory).
    simulation_pool_config : SimulationPoolConfig
        Configuration for each SimulationAgent.
    training_agent_config : TrainingAgentConfig | None
        Configuration for the TrainingAgent. If None, training is disabled
        and no CVAE model updates will occur.
    inference_agent_config : InferenceAgentConfig | None
        Configuration for the InferenceAgent. If None, the legacy
        OrchestratorAgent-based iteration loop is used instead.
    analysis_pool_config : AnalysisPoolConfig | None
        Configuration for the legacy analysis pool agent (Phase 3).
        Only used when inference_agent_config is None.
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
    num_simulations: int = Field(
        default=4,
        ge=1,
        description='Number of parallel SimulationAgents to launch.',
    )
    simulation_pool_config: SimulationPoolConfig = Field(
        description='Configuration for each SimulationAgent.',
    )
    training_agent_config: TrainingAgentConfig | None = Field(
        default=None,
        description='Configuration for the TrainingAgent. '
        'If None, CVAE training is disabled.',
    )
    inference_agent_config: InferenceAgentConfig | None = Field(
        default=None,
        description='Configuration for the InferenceAgent. '
        'If None, the legacy OrchestratorAgent loop is used.',
    )
    analysis_pool_config: AnalysisPoolConfig | None = Field(
        default=None,
        description='Configuration for the legacy analysis pool (Phase 3).',
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

