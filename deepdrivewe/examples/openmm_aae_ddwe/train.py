"""Training module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from deepdrivewe import SimResult
from deepdrivewe import TrainResult
from deepdrivewe import validate_and_resolve_file
from deepdrivewe.ai import AdversarialAE
from deepdrivewe.ai import AdversarialAEConfig


class TrainConfig(BaseModel):
    """Arguments for the training module."""

    config_path: Path = Field(
        description='The path to the model configuration file.',
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description='The path to the model checkpoint file.'
        'Train from scratch by default.',
    )

    @field_validator('config_path', 'checkpoint_path')
    @classmethod
    def validate_and_resolve_file(cls, value: Path | None) -> Path | None:
        """Validate and resolve the file path."""
        return validate_and_resolve_file(value)


# TODO: We probably need to store a history of old training data
# to retrain the model. Add a config argument to include a cMD run dataset.
# Contact maps: https://github.com/n-frazee/DL-enhancedWE/blob/main/common_files/train.npy


def run_train(
    sim_output: list[SimResult],
    config: TrainConfig,
    output_dir: Path,
) -> TrainResult:
    """Train the model on the simulation output."""
    # Make the output directory
    itetation = sim_output[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model configuration
    model_config = AdversarialAEConfig.from_yaml(config.config_path)

    # Load the model
    model = AdversarialAE(
        model_config,
        checkpoint_path=config.checkpoint_path,
    )

    # Extract the last frame contact maps and rmsd from each simulation
    coordinates = np.concatenate(
        [sim.data['coordinates'] for sim in sim_output],
    )
    pcoords = np.concatenate([sim.data['pcoords'] for sim in sim_output])
    pcoords = pcoords.flatten()

    # Fit the model
    checkpoint_path = model.fit(
        x=coordinates,
        model_dir=output_dir / 'model',
        scalars={'pcoord': pcoords},
    )

    # Return the train result
    result = TrainResult(
        config_path=config.config_path,
        checkpoint_path=checkpoint_path,
    )

    return result
