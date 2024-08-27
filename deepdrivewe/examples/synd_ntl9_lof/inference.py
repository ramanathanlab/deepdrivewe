"""Inference module for the synD LOF example."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from pydantic import Field
from sklearn.neighbors import LocalOutlierFactor

from deepdrivewe import BasisStates
from deepdrivewe import IterationMetadata
from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe import TargetState
from deepdrivewe.ai import ConvolutionalVAE
from deepdrivewe.ai import ConvolutionalVAEConfig
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import LOFLowResampler


class InferenceConfig(BaseModel):
    """Arguments for the inference module."""

    # AI model settings
    ai_model_config_path: Path = Field(
        description='The path to the CVAE model YAML configuration file.',
    )
    ai_model_checkpoint_path: Path = Field(
        description='The path to the CVAE model checkpoint file.',
    )
    ai_model_latent_history_path: Path | None = Field(
        default=None,
        description='The path to the latent space history file (z.npy).'
        'A small batch of latent coordinates used to provide context for LOF.',
    )
    ai_model_inference_batch_size: int = Field(
        default=128,
        description='The batch size for inference.',
    )

    # Local outlier factor settings
    lof_n_neighbors: int = Field(
        default=20,
        description='The number of neighbors to use for LOF.',
    )
    lof_distance_metric: str = Field(
        default='cosine',
        description='The distance metric to use for LOF [cosine, minkowski].',
    )

    # Resampling settings
    sims_per_bin: int = Field(
        default=72,
        description='The number of simulations to maintain in each bin.'
        ' Default is 72.',
    )
    consider_for_resampling: int = Field(
        default=12,
        description='The number of simulations to consider for resampling.',
    )
    max_resamples: int = Field(
        default=4,
        description='The maximum number of resamples to perform in each '
        'iteration. Default is 4.',
    )
    max_allowed_weight: float = Field(
        default=1.0,
        description='The maximum allowed weight for a simulation. Default '
        'is 1.0.',
    )
    min_allowed_weight: float = Field(
        default=10e-40,
        description='The minimum allowed weight for a simulation. Default '
        'is 10e-40.',
    )


# TODO: We are not checkpointing the latent space history. This is
# may cause an issue if we find the latent history is very important
# for convergence and we have a restart. We should consider checkpointing.
class LatentSpaceHistory:
    """A class to store the latent space history."""

    def __init__(self) -> None:
        """Initialize the latent space history."""
        # A (n, d) array where n is the number of frames and d is
        # the latent space dimensionality
        self.z = np.array([])
        # A (n, 1) array of progress coordinates for each frame
        self.pcoords = np.array([])

    def __bool__(self) -> bool:
        """Return True if the history is not empty."""
        return bool(len(self.z))

    def update(self, z: ArrayLike, pcoords: ArrayLike) -> None:
        """Update the latent space history.

        Parameters
        ----------
        z: array-like
            The latent space coordinates (n_frames, d)
        pcords: array-like
            The progress coordinates (n_frames, 1)
        """
        self.z = z
        self.pcoords = pcoords

    def plot(
        self,
        output_path: Path,
        color: ArrayLike | None = None,
        cblabel: str = 'Progress Coordinate',
        title: str = '',
    ) -> None:
        """Create a scatter plot.

        Parameters
        ----------
        x: array-like
            X data for the scatter plot
        y: array-like
            Y data for the scatter plot
        color: array-like
            Data used for coloring the points
        xlabel: str
            Label for the X-axis
        ylabel: str
            Label for the Y-axis
        title: str
            Title of the plot
        """
        import matplotlib.pyplot as plt

        # Set the color data to the progress coordinates if not provided
        color = self.pcoords if color is None else color

        print(
            f'Plotting latent space to with {len(self.z)} '
            f'and color with shape {len(color)} frames to {output_path}',
        )

        # Create the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            xs=self.z[:, 0],
            ys=self.z[:, 1],
            zs=self.z[:, 2],
            c=color,
            cmap='viridis',
        )
        fig.colorbar(scatter, label=cblabel)
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.set_zlabel(r'$z_3$')
        ax.set_title(title)
        # plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.savefig(output_path)
        plt.close()


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


def run_inference(
    input_data: list[SimResult],
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
    output_dir: Path,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Make the output directory
    itetation = input_data[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the rmsd pcoord from the last frame of each simulation
    pcoords = [sim.metadata.pcoord[-1][0] for sim in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    cur_sims = [sim.metadata for sim in input_data]

    # Load the model and history
    model, history = warmstart_model(
        config.ai_model_config_path,
        config.ai_model_checkpoint_path,
    )

    # Extract the last frame contact maps and rmsd from each simulation
    contact_maps = [sim.data['contact_maps'][-1] for sim in input_data]
    pcoords = [sim.data['pcoords'][-1] for sim in input_data]

    # Convert to int16
    contact_maps = [x.astype(np.int16) for x in contact_maps]

    # Compute the latent space representation
    z = model.predict(
        x=contact_maps,
        inference_batch_size=config.ai_model_inference_batch_size,
    )

    # Concatenate the latent history
    if history:
        z = np.concatenate([history.z, z])
        pcoords = np.concatenate([history.pcoords, pcoords])

    # Run LOF on the latent space
    clf = LocalOutlierFactor(
        n_neighbors=config.lof_n_neighbors,
        metric=config.lof_distance_metric,
    ).fit(z)

    # Get the LOF scores
    lof_scores = clf.negative_outlier_factor_

    # Update the latent space history
    history.update(z, pcoords)

    # Plot the latent space
    history.plot(output_dir / 'pcoord.png')
    history.plot(
        output_dir / 'pcoord_lof.png',
        color=lof_scores,
        cblabel='LOF Score',
    )

    # Add the LOF scores to the last frame of each simulation pcoord
    for sim, score in zip(cur_sims, lof_scores[-len(cur_sims) :]):
        sim_scores = [-1.0 for _ in range(sim.num_frames)]
        sim_scores[-1] = float(score)
        sim.append_pcoord(sim_scores)

    # Create the binner
    binner = RectilinearBinner(
        bins=[0.0, 1.0, float('inf')],
        bin_target_counts=config.sims_per_bin,
    )

    # Define the recycling policy
    recycler = LowRecycler(
        basis_states=basis_states,
        target_threshold=target_states[0].pcoord[0],
    )

    # Resamlpe the ensemble
    resampler = LOFLowResampler(
        consider_for_resampling=config.consider_for_resampling,
        max_resamples=config.max_resamples,
        max_allowed_weight=config.max_allowed_weight,
        min_allowed_weight=config.min_allowed_weight,
    )

    # Assign simulations to bins and resample the weighted ensemble
    result = resampler.run(cur_sims, binner, recycler)

    return result
