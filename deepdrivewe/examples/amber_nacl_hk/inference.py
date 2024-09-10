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
from deepdrivewe.resamplers import HuberKimResampler


class InferenceConfig(BaseModel):
    """Arguments for the naive resampler."""

    sims_per_bin: int = Field(
        default=5,
        description='The number of simulations maintain in each bin.'
        ' Default is 5.',
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
    basis_states: BasisStates,
    target_states: list[TargetState],
    config: InferenceConfig,
) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.metadata.pcoord[-1] for sim_result in sim_output]

    print(f'Progress coordinates: {pcoords}')
    print(f'Best progress coordinate: {min(pcoords)}')
    print(f'Num input simulations: {len(sim_output)}')

    # Extract the simulation metadata
    cur_sims = [sim_result.metadata for sim_result in sim_output]

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

    # Define the resampling policy
    resampler = HuberKimResampler(
        sims_per_bin=config.sims_per_bin,
        max_allowed_weight=config.max_allowed_weight,
        min_allowed_weight=config.min_allowed_weight,
    )

    # Assign simulations to bins and resample the weighted ensemble
    result = resampler.run(cur_sims, binner, recycler)

    return result
