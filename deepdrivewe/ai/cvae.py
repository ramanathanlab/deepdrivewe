"""Convolutional Variational Autoencoder for Contact Maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from pydantic import Field

from deepdrivewe import BaseModel


class ConvolutionalVAEConfig(BaseModel):
    """Settings for mdlearn SymmetricConv2dVAETrainer."""

    input_shape: tuple[int, int, int] = Field(
        default=(1, 40, 40),
        description='The shape of the input contact maps.',
    )
    filters: list[int] = Field(
        default=[16, 16, 16, 16],
        description='The number of filters in each convolutional layer.',
    )
    kernels: list[int] = Field(
        default=[3, 3, 3, 3],
        description='The kernel size in each convolutional layer.',
    )
    strides: list[int] = Field(
        default=[1, 1, 1, 2],
        description='The stride in each convolutional layer.',
    )
    affine_widths: list[int] = Field(
        default=[128],
        description='The width of the affine layers.',
    )
    affine_dropouts: list[float] = Field(
        default=[0.5],
        description='The dropout rate for the affine layers.',
    )
    latent_dim: int = Field(
        default=3,
        description='The dimensionality of the latent space.',
    )
    lambda_rec: float = Field(
        default=1.0,
        description='The reconstruction loss weight.',
    )
    num_data_workers: int = Field(
        default=0,
        description='The number of data workers for the data loader.',
    )
    prefetch_factor: int | None = Field(
        default=None,
        description='The prefetch factor for the data loader.',
    )
    batch_size: int = Field(
        default=64,
        description='The batch size for training.',
    )
    device: str = Field(
        default='cuda',
        description='The device to use for training.',
    )
    optimizer_name: str = Field(
        default='RMSprop',
        description='The optimizer to use for training.',
    )
    optimizer_hparams: dict[str, float] = Field(
        default={
            'lr': 0.001,
            'weight_decay': 0.00001,
        },
        description='The hyperparameters for the optimizer.',
    )
    epochs: int = Field(
        default=100,
        description='The number of epochs to train for.',
    )
    checkpoint_log_every: int = Field(
        default=25,
        description='The number of epochs between checkpoint saves.',
    )
    plot_log_every: int = Field(
        default=25,
        description='The number of epochs between plot saves.',
    )
    plot_n_samples: int = Field(
        default=5000,
        description='The number of samples to plot.',
    )
    plot_method: str | None = Field(
        default='raw',
        description='The method to use for plotting.',
    )


class ConvolutionalVAE:
    """A convolutional variational autoencoder for contact maps."""

    def __init__(
        self,
        config: ConvolutionalVAEConfig,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Initialize the ConvolutionalVAE.

        Parameters
        ----------
        config : ConvolutionalVAEConfig
            The configuration settings for the VAE.
        checkpoint_path : Path, optional
            The path to the model checkpoint to load, by default None.
        """
        # Lazy import to avoid needing torch to load module
        from mdlearn.nn.models.vae.symmetric_conv2d_vae import (
            SymmetricConv2dVAETrainer,
        )

        self.config = config
        self.checkpoint_path = checkpoint_path
        self.trainer = SymmetricConv2dVAETrainer(**self.config.dict())

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
        z, *_ = self.trainer.predict(x)
        return z
