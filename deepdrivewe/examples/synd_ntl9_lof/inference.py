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
from deepdrivewe.resamplers import LOFLowResampler
from deepdrivewe.workflows.utils import batch_data


class InferenceConfig(BaseModel):
    """Arguments for the inference module."""

    # AI model settings
    model_config_path: Path = Field(
        description='The path to the CVAE model YAML configuration file.',
    )
    model_checkpoint_path: Path = Field(
        description='The path to the CVAE model checkpoint file.',
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
    max_resamples: int = Field(
        default=4,
        description='The maximum number of resamples to perform in each '
        'iteration. Default is 4.',
    )


def run_inference(
    input_data: list[SimResult],
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim.metadata.pcoord[-1] for sim in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    cur_sims = [sim.metadata for sim in input_data]

    # Load the model configuration
    model_config = ConvolutionalVAEConfig.from_yaml(config.model_config_path)

    # Load the model
    model = ConvolutionalVAE(
        model_config,
        checkpoint_path=config.model_checkpoint_path,
    )

    # TODO: We may need to keep embedding data from all the iterations
    # to compute reliable LOF scores.

    # Extract the contact maps from each simulation
    data = [sim.data for sim in input_data]
    contact_maps = np.array([x['contact_map'] for x in data])

    # Compute the latent space representation
    z = model.predict(x=contact_maps)

    # Run LOF on the latent space
    clf = LocalOutlierFactor(
        n_neighbors=config.lof_n_neighbors,
        metric=config.lof_distance_metric,
    ).fit(z)

    # Get the LOF scores
    lof_scores = clf.negative_outlier_factor_.tolist()

    # Group the simulations by LOF score
    sim_scores = batch_data(lof_scores, batch_size=len(cur_sims))

    # Check that the number of simulations matches the batched scores
    assert len(sim_scores) == len(cur_sims)

    # Loop over each simulation and add the LOF scores
    for sim, scores in zip(cur_sims, sim_scores):
        sim.append_pcoord(scores)

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
        consider_for_resampling=config.sims_per_bin,
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
