"""Inference module for the basic_amber example."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from deepdrivewe import BasisStates
from deepdrivewe import IterationMetadata
from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe import TargetState
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import SplitLowResampler


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
    total_simulations: int = Field(
        default=10,
        description='The total number of simulations to maintain in the'
        ' ensemble. Default is 10.',
    )


def run_inference(
    input_data: list[SimResult],
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.metadata.pcoord for sim_result in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    cur_sims = [sim_result.metadata for sim_result in input_data]

    # Create the binner
    binner = RectilinearBinner(
        bins=[
            float('-inf'),
            2.6,
            float('inf'),
        ],
        bin_target_counts=config.total_simulations,
    )

    # Define the recycling policy
    recycler = LowRecycler(
        basis_states=basis_states,
        target_threshold=target_states[0].pcoord[0],
    )

    # Resamlpe the ensemble
    resampler = SplitLowResampler(
        num_resamples=config.num_resamples,
        n_split=config.n_split,
    )

    # Get the next iteration of simulation metadata
    next_sims = resampler.get_next_sims(cur_sims)

    # Recycle the current iteration
    cur_sims, next_sims = recycler.recycle_simulations(cur_sims, next_sims)

    # Compute the iteration metadata
    metadata = binner.compute_iteration_metadata(cur_sims)

    cur_sims, next_sims = resampler.resample(cur_sims, next_sims)

    return cur_sims, next_sims, metadata
