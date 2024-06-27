"""Inference module for the basic_amber example."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.binning import RectilinearBinner
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import SimMetadata
from westpa_colmena.examples.amber_split_low.simulate import SimResult
from westpa_colmena.resampling import LowRecycler
from westpa_colmena.resampling import SplitLowResampler


class InferenceConfig(BaseModel):
    """Arguments for the naive resampler."""

    num_resamples: int = Field(
        default=1,
        description='The number of resamples to perform (i.e., the number of'
        ' splits and merges to perform in each iteration). Default is 1.',
    )
    n_split: int = Field(
        default=2,
        description='The number of simulations to split each simulation into.'
        ' Default is 2.',
    )
    target_threshold: float = Field(
        default=0.5,
        description='The target threshold for the progress coordinate to be'
        ' considered in the target state. Default is 0.5.',
    )


def run_inference(
    input_data: list[SimResult],
    basis_states: BasisStates,
    config: InferenceConfig,
) -> list[SimMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.metadata.pcoord for sim_result in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')  # type: ignore[type-var]
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    current_iteration = [sim_result.metadata for sim_result in input_data]

    # Define the recycling policy
    recycler = LowRecycler(target_threshold=config.target_threshold)

    # Resamlpe the ensemble
    resampler = SplitLowResampler(
        basis_states=basis_states,
        recycler=recycler,
        num_resamples=config.num_resamples,
        n_split=config.n_split,
    )

    # Get the next iteration
    next_iteration = resampler.get_next_iteration(current_iteration)

    # Create the binner
    binner = RectilinearBinner(
        resampler=resampler,
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

    # Assign the simulations to bins
    binned_sims = binner.assign(next_iteration)

    # Get the next iteration of simulations
    new_sims = []
    for sims in binned_sims:
        new_sims.extend(resampler.resample(sims))

    return new_sims
