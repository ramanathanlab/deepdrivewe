"""Training module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from deepdrivewe import SimResult
from deepdrivewe import TrainResult
from deepdrivewe.ai import ConvolutionalVAE
from deepdrivewe.ai import ConvolutionalVAEConfig

from proxystore.store import get_store

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


# TODO: We probably need to store a history of old training data
# to retrain the model. Add a config argument to include a cMD run dataset.
# Contact maps: https://github.com/n-frazee/DL-enhancedWE/blob/main/common_files/train.npy


def run_train(
    sim_output: list, # List of raw ProxyStore Keys
    config: TrainConfig,
    output_dir: Path,
) -> TrainResult:
    """Train the model on the simulation output."""
    # Make the output directory
    itetation = sim_output[0].metadata.iteration_id

    # Manually resolve the keys using the registered 'file-store'
    store = get_store('file-store')
    if store is None:
        raise RuntimeError("ProxyStore 'file-store' is not initialized on the worker.")

    # store.get(key) retrieves the object without the destructive 'evict' behavior
    resolved_sims = [store.get(key) for key in sim_output]
    print(f"DEBUG: Successfully resolved {len(resolved_sims)} simulation objects", flush=True)

    # Make the output directory using the first resolved object
    iteration = resolved_sims[0].metadata.iteration_id
    output_dir = output_dir / f'{iteration:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract contact maps and pcoords from the resolved objects
    contact_maps = np.concatenate(
        [sim.data['contact_maps'] for sim in resolved_sims],
        axis=0 # join along the frame/sample axis
    )
    pcoords = np.concatenate([sim.data['pcoords'] for sim in resolved_sims])
    pcoords = pcoords.flatten()

    # Load the model configuration
    model_config = ConvolutionalVAEConfig.from_yaml(config.config_path)

    # Load the model
    model = ConvolutionalVAE(
        model_config,
        checkpoint_path=config.checkpoint_path,
    )

    # Fit the model
    checkpoint_path = model.fit(
        x=contact_maps,
        model_dir=output_dir / 'model',
        scalars={'pcoord': pcoords},
    )

    # Return the train result
    result = TrainResult(
        config_path=config.config_path,
        checkpoint_path=checkpoint_path,
    )

    return result
