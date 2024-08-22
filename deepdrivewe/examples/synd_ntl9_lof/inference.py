"""Inference module for the synD LOF example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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

# from deepdrivewe.resamplers import LOFLowResampler
from deepdrivewe.resamplers.lof_v2 import LOFLowResamplerV2
from deepdrivewe.workflows.utils import batch_data


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


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    color_data: np.ndarray,
    output_path: Path,
    xlabel: str = 'X-axis',
    ylabel: str = 'Y-axis',
    title: str = '',
) -> None:
    """Create a scatter plot.

    Parameters
    ----------
    x: array-like
        X data for the scatter plot
    y: array-like
        Y data for the scatter plot
    color_data: array-like
        Data used for coloring the points
    filename: str
        Filename for the saved image (default: 'scatter_plot.png')
    xlabel: str
        Label for the X-axis
    ylabel: str
        Label for the Y-axis
    title: str
        Title of the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=color_data, cmap='viridis')
    plt.colorbar(scatter, label='Color Intensity')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def run_inference(  # noqa: PLR0915
    input_data: list[SimResult],
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
    output_dir: Path,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Extract the rmsd pcoord from the last frame of each simulation
    pcoords = [sim.metadata.pcoord[-1][0] for sim in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    cur_sims = [sim.metadata for sim in input_data]

    # Load the model configuration
    model_config = ConvolutionalVAEConfig.from_yaml(
        config.ai_model_config_path,
    )

    # Load the model
    model = ConvolutionalVAE(
        model_config,
        checkpoint_path=config.ai_model_checkpoint_path,
    )

    # Extract the contact maps from each simulation
    data = [sim.data for sim in input_data]
    contact_maps = []
    for sim_data in data:
        contact_maps.extend(sim_data['contact_maps'].astype(np.int16))
    # contact_maps = np.array(cmaps, dtype=object)

    print(f'{len(contact_maps)=}')
    print(f'{contact_maps[0].shape=}')
    print(f'{contact_maps[0]=}')
    print(f'{contact_maps[1]=}')
    # print(f'{contact_maps.shape=}', flush=True)

    # Compute the latent space representation
    z = model.predict(x=contact_maps)

    # Load the latent history if provided
    if config.ai_model_latent_history_path is not None:
        z_history = np.load(config.ai_model_latent_history_path)
        z = np.concatenate([z, z_history])

    # Run LOF on the latent space
    clf = LocalOutlierFactor(
        n_neighbors=config.lof_n_neighbors,
        metric=config.lof_distance_metric,
    ).fit(z)

    # Get the LOF scores
    lof_scores = clf.negative_outlier_factor_.tolist()

    if config.ai_model_latent_history_path is not None:
        # Remove the history from the LOF scores
        lof_scores = lof_scores[: -z_history.shape[0]]
        z = z[: -z_history.shape[0]]

    # Group the simulations by LOF score
    sim_scores = batch_data(
        lof_scores,
        batch_size=len(lof_scores) // len(cur_sims),
    )

    print(f'{len(sim_scores)=}')
    print(f'{len(cur_sims)=}', flush=True)

    # Check that the number of simulations matches the batched scores
    assert len(sim_scores) == len(cur_sims)
    # Check that there is one LOF score per simulation frame
    assert len(sim_scores[0]) == len(cur_sims[0].pcoord)

    # Loop over each simulation and add the LOF scores
    for sim, scores in zip(cur_sims, sim_scores):
        sim.append_pcoord(scores)

    # Log the latent space
    all_pcoords = [sim.metadata.pcoord for sim in input_data]
    pcoords = np.array(
        [frame[0] for sim in all_pcoords for frame in sim],
    ).reshape(-1, 1)

    itetation = input_data[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'z.npy', z)
    np.save(output_dir / 'lof.npy', lof_scores)
    np.save(output_dir / 'pcoord.npy', pcoords)
    scatter_plot(z[:, 0], z[:, 1], lof_scores, output_dir / 'lof.png')
    scatter_plot(z[:, 0], z[:, 1], pcoords, output_dir / 'pcoord.png')

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
    resampler = LOFLowResamplerV2(
        consider_for_resampling=config.consider_for_resampling,
        max_resamples=config.max_resamples,
        max_allowed_weight=config.max_allowed_weight,
        min_allowed_weight=config.min_allowed_weight,
    )

    # Get the next iteration of simulation metadata
    next_sims = resampler.get_next_sims(cur_sims)

    # Recycle the current iteration
    cur_sims, next_sims = recycler.recycle_simulations(cur_sims, next_sims)

    # Assign the simulations to bins
    bin_assignments = binner.bin_simulations(next_sims)

    # Compute the iteration metadata
    metadata = binner.compute_iteration_metadata(cur_sims)

    # Resample the simulations in each bin
    new_sims = []
    for bin_sims in bin_assignments.values():
        # Get the simulations in the bin
        binned_sims = [next_sims[sim_idx] for sim_idx in bin_sims]

        # Resample the bin and add them to the new simulations
        cur_sims, resampled_sims = resampler.resample(cur_sims, binned_sims)

        # Add the resampled simulations to the new simulations
        new_sims.extend(resampled_sims)

    return cur_sims, new_sims, metadata
