"""Inference module for the LOF strategy."""

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
from deepdrivewe import TrainResult
from deepdrivewe.ai import warmstart_model
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import LOFLowResampler


class InferenceConfig(BaseModel):
    """Arguments for the inference module."""

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


def run_inference(
    sim_output: list[SimResult],
    train_output: TrainResult,
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
    output_dir: Path,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Make the output directory
    itetation = sim_output[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the rmsd pcoord from the last frame of each simulation
    pcoords = [sim.metadata.pcoord[-1][0] for sim in sim_output]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(sim_output)}')

    # Extract the simulation metadata
    cur_sims = [sim.metadata for sim in sim_output]

    # Load the model and history
    model, history = warmstart_model(
        train_output.config_path,
        train_output.checkpoint_path,
    )

    # Extract the last frame contact maps and rmsd from each simulation
    contact_maps = [sim.data['contact_maps'][-1] for sim in sim_output]
    pcoords = [sim.data['pcoords'][-1] for sim in sim_output]

    # Convert to int16
    contact_maps = [x.astype(np.int16) for x in contact_maps]

    # Compute the latent space representation
    z = model.predict(x=contact_maps)

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

    # Define the resampling policy
    resampler = LOFLowResampler(
        consider_for_resampling=config.consider_for_resampling,
        max_resamples=config.max_resamples,
        max_allowed_weight=config.max_allowed_weight,
        min_allowed_weight=config.min_allowed_weight,
    )

    # Assign simulations to bins and resample the weighted ensemble
    result = resampler.run(cur_sims, binner, recycler)

    return result
