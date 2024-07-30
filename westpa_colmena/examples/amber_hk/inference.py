"""Inference module for the basic_amber example."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.binning import RectilinearBinner
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import IterationMetadata
from westpa_colmena.ensemble import SimMetadata
from westpa_colmena.ensemble import TargetState
from westpa_colmena.examples.amber_hk.simulate import SimResult
from westpa_colmena.recycling import LowRecycler
from westpa_colmena.resampling import HuberKimResampler


class InferenceConfig(BaseModel):
    """Arguments for the naive resampler."""

    sims_per_bin: int = Field(
        default=5,
        description='The number of simulations maintain in each bin.'
        ' Default is 5.',
    )
    max_allowed_weight: float = Field(
        default=0.25,
        description='The maximum allowed weight for a simulation. Default '
        'is 0.25.',
    )
    min_allowed_weight: float = Field(
        default=10e-40,
        description='The minimum allowed weight for a simulation. Default '
        'is 10e-40.',
    )


def run_inference(
    input_data: list[SimResult],
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.metadata.pcoord[-1] for sim_result in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    cur_sims = [sim_result.metadata for sim_result in input_data]

    # Create the binner
    binner = RectilinearBinner(
        bins=[
            0.00,
            2.60,
            2.80,
            3.00,
            3.20,
            3.40,
            3.60,
            3.80,
            4.00,
            4.50,
            5.00,
            5.50,
            6.00,
            7.00,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            float('inf'),
        ],
        bin_target_counts=config.sims_per_bin,
    )

    # Define the recycling policy
    recycler = LowRecycler(
        basis_states=basis_states,
        target_threshold=target_states[0].pcoord[0],
    )

    # Resamlpe the ensemble
    resampler = HuberKimResampler(
        sims_per_bin=config.sims_per_bin,
        max_allowed_weight=config.max_allowed_weight,
        min_allowed_weight=config.min_allowed_weight,
    )

    # Get the next iteration of simulation metadata
    next_sims = resampler.get_next_iteration(cur_sims)

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
