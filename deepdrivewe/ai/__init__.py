"""Module for artificial intelligence methods."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Forward imports
from deepdrivewe.ai.cvae import ConvolutionalVAE
from deepdrivewe.ai.cvae import ConvolutionalVAEConfig
from deepdrivewe.ai.utils import LatentSpaceHistory


@lru_cache(maxsize=1)
def warmstart_model(
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[ConvolutionalVAE, LatentSpaceHistory]:
    """Load the model once and then return a cached version.

    Parameters
    ----------
    config_path : Path
        The path to the model configuration file.
    checkpoint_path : Path
        The path to the model checkpoint file.

    Returns
    -------
    ConvolutionalVAE
        The ConvolutionalVAE model.
    LatentSpaceHistory
        The latent space history.
    """
    # Print the warmstart message
    print(f'Cold start model from checkpoint {checkpoint_path}')

    # Load the model configuration
    model_config = ConvolutionalVAEConfig.from_yaml(config_path)

    # Load the model
    model = ConvolutionalVAE(
        model_config,
        checkpoint_path=checkpoint_path,
    )

    # Initialize the latent space history
    history = LatentSpaceHistory()

    return model, history
