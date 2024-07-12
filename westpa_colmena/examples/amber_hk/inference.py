"""Inference module for the basic_amber example."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.binning import RectilinearBinner
from westpa_colmena.ensemble import BasisStates
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
) -> tuple[list[SimMetadata], list[list[SimMetadata]]]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.metadata.pcoord for sim_result in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')  # type: ignore[type-var]
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    current_iteration = [sim_result.metadata for sim_result in input_data]

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

    # TODO: When we log the results to HDF5 we want the current iteration
    #       before recycling.

    # Recycle the current iteration
    current_iteration = recycler.recycle(current_iteration)

    # Get the next iteration of simulation metadata
    next_iteration = resampler.get_next_iteration(current_iteration)

    # Assign the simulations to bins
    binned_sims = binner.bin_simulations(next_iteration)

    # Resample the next iteration
    new_sims = []
    for sims in binned_sims:
        new_sims.extend(resampler.resample(sims))

    return new_sims, binned_sims
