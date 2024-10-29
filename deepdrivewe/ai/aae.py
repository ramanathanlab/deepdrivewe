"""Adversarial Autoencoder for Contact Maps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe.ai.utils import LatentSpaceHistory


class AdversarialAEConfig(BaseModel):
    """Settings for mdlearn 3dAAE model."""

    scalar_dset_names: list[str] = Field(
        description='Name of scalar datasets to paint w.r.t.',
    )
    num_points: int = Field(
        default=3378,  # Number of Spike protein residues
        description='Number of residues in the protein'
        ' (i.e., points in the point cloud).',
    )
    num_features: int = Field(
        default=0,
        description='Number of additional per-point features'
        ' in addition to xyz coords.',
    )
    latent_dim: int = Field(
        default=3,
        description='Dimensionality of the latent space.',
    )
    encoder_bias: bool = Field(
        default=True,
        description='Whether to use bias in the encoder.',
    )
    encoder_relu_slope: float = Field(
        default=0.0,
        description='The slope of the ReLU function in the encoder.',
    )
    encoder_filters: list[int] = Field(
        default=[64, 128, 256, 256, 512],
        description='The number of filters in each convolutional layer'
        ' of the encoder.',
    )
    encoder_kernels: list[int] = Field(
        default=[5, 3, 3, 1, 1],
        description='The kernel size in each convolutional layer '
        'of the encoder.',
    )
    decoder_bias: bool = Field(
        default=True,
        description='Whether to use bias in the decoder.',
    )
    decoder_relu_slope: float = Field(
        default=0.0,
        description='The slope of the ReLU function in the decoder.',
    )
    decoder_affine_widths: list[int] = Field(
        default=[64, 128, 512, 1024],
        description='The width of the affine layers in the decoder.',
    )
    discriminator_bias: bool = Field(
        default=True,
        description='Whether to use bias in the discriminator.',
    )
    discriminator_relu_slope: float = Field(
        default=0.0,
        description='The slope of the ReLU function in the discriminator.',
    )
    discriminator_affine_widths: list[int] = Field(
        default=[512, 512, 128, 64],
        description='The width of the affine layers in the discriminator.',
    )
    noise_mu: float = Field(
        default=0.0,
        description='Mean of the prior distribution.',
    )
    noise_std: float = Field(
        default=0.2,
        description='Standard deviation of the prior distribution.',
    )
    lambda_gp: float = Field(
        default=10.0,
        description='Relative weight to put on gradient penalty.',
    )
    lambda_rec: float = Field(
        default=0.5,
        description='Relative weight to put on reconstruction loss.',
    )
    num_data_workers: int = Field(
        default=0,
        description='Number of data loaders for inference.',
    )
    batch_size: int = Field(
        default=32,
        description='Inference batch size.',
    )
    inference_batch_size: int = Field(
        default=64,
        description='Inference batch size.',
    )


class AdversarialAE:
    """Adversarial autoencoder for protein conformers."""

    def __init__(
        self,
        config: AdversarialAEConfig,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Initialize the ConvolutionalVAE.

        Parameters
        ----------
        config : AdversarialAEConfig
            The configuration settings for the model.
        checkpoint_path : Path, optional
            The path to the model checkpoint to load, by default None.
        """
        # Lazy import to avoid needing torch to load module
        from mdlearn.nn.models.aae.point_3d_aae import AAE3dTrainer

        self.config = config
        self.checkpoint_path = checkpoint_path

        # We keep the inference_batch_size in the config for convenience
        # but exclude it from the model arguments.
        model_args = config.model_dump(exclude={'inference_batch_size'})
        self.trainer = AAE3dTrainer(**model_args)

        # Load the model checkpoint if specified
        if checkpoint_path is not None:
            self.update_model(checkpoint_path)

    def update_model(self, checkpoint_path: Path) -> None:
        """Update the model with a new checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            The path to the checkpoint to load.
        """
        # Skip if the checkpoint path is the same
        if checkpoint_path == self.checkpoint_path:
            return

        # Lazy import to avoid needing torch to load module
        import torch

        # Load the checkpoint
        cp = torch.load(checkpoint_path, map_location=self.trainer.device)

        # Load the model state dict
        self.trainer.model.load_state_dict(cp['model_state_dict'])

        # Update the checkpoint path
        self.checkpoint_path = checkpoint_path

    def fit(
        self,
        x: np.ndarray,
        model_dir: Path,
        scalars: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Fit the model to the input data.

        Parameters
        ----------
        x : np.ndarray
            The contact maps to fit the model to. (n_samples, *) where * is a
            ragged dimension containing the concatenated row and column indices
            of the ones in the contact map.
        model_dir : Path
            The directory to save the model to.
        scalars : dict[str, np.ndarray], optional
            The scalars to plot during training, by default None.

        Returns
        -------
        Path
            The path to the most recent model checkpoint.
        """
        # Setup the scalars for plotting if specified
        scalars = {} if scalars is None else scalars

        # Fit the model
        self.trainer.fit(X=x, scalars=scalars, output_path=model_dir)

        # Log the loss curve to a CSV file
        pd.DataFrame(self.trainer.loss_curve_).to_csv(model_dir / 'loss.csv')

        # Get the most recent model checkpoint from the checkpoint directory
        checkpoint_dir = model_dir / 'checkpoints'
        checkpoint_path = natsorted(list(checkpoint_dir.glob('*.pt')))[-1]

        return checkpoint_path

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the latent space coordinates for a given set of coordinates.

        Parameters
        ----------
        x: np.ndarray
            The contact maps to predict the latent space coordinates for
            (n_samples, *) where * is a ragged dimension containing the
            concatenated row and column indices of the ones in the contact map.

        Returns
        -------
        np.ndarray
            The predicted latent space coordinates (n_samples, latent_dim).
        """
        # Predict the latent space coordinates
        z, _ = self.trainer.predict(x, self.config.inference_batch_size)
        return z


@lru_cache(maxsize=1)
def warmstart_aae(
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[AdversarialAE, LatentSpaceHistory]:
    """Load the model once and then return a cached version.

    Parameters
    ----------
    config_path : Path
        The path to the model configuration file.
    checkpoint_path : Path
        The path to the model checkpoint file.

    Returns
    -------
    AdversarialAE
        The AdversarialAE model.
    LatentSpaceHistory
        The latent space history.
    """
    # Print the warmstart message
    print(f'Cold start model from checkpoint {checkpoint_path}')

    # Load the model configuration
    model_config = AdversarialAEConfig.from_yaml(config_path)

    # Load the model
    model = AdversarialAE(
        model_config,
        checkpoint_path=checkpoint_path,
    )

    # Initialize the latent space history
    history = LatentSpaceHistory()

    return model, history
