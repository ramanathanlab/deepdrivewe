"""Inference module for the basic_amber example."""

from __future__ import annotations

from westpa_colmena.ensemble import SimulationMetadata
from westpa_colmena.examples.basic_amber.simulate import SimulationResult


def run_inference(
    input_data: list[SimulationResult],
) -> list[SimulationMetadata]:
    """Run inference on the input data."""
    from westpa_colmena.ensemble import NaiveResampler

    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.pcoord[-1] for sim_result in input_data]

    # Extract the simulation metadata
    current_iteration = [sim_result.metadata for sim_result in input_data]

    # Resamlpe the ensemble
    resampler = NaiveResampler(pcoord=pcoords)

    # Get the next iteration of simulations
    next_iteration = resampler.resample(current_iteration)

    return next_iteration
