"""Inference module for the basic_amber example."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.ensemble import Resampler
from westpa_colmena.ensemble import SimulationMetadata
from westpa_colmena.examples.basic_amber.simulate import SimulationResult


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
    split_low: bool = Field(
        default=True,
        description='If True, split the simulation with the lowest progress'
        ' coordinate and merge the simulations with the highest progress'
        ' coordinate. If False, split the simulation with the highest'
        ' progress coordinate and merge the simulations with the lowest'
        ' progress coordinate. Default is True.',
    )
    target_threshold: float = Field(
        default=0.5,
        description='The target threshold for the progress coordinate to be'
        ' considered in the target state. Default is 0.5.',
    )


class NaiveResampler(Resampler):
    """Naive resampler."""

    def __init__(  # noqa: PLR0913
        self,
        pcoord: list[float],
        num_resamples: int = 1,
        n_split: int = 2,
        split_low: bool = True,
        target_threshold: float = 0.5,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        pcoord : list[float]
            The progress coordinate for the simulations.
        num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        n_split : int
            The number of simulations to split each simulation into.
            Default is 2.
        split_low : bool
            If True, split the simulation with the lowest progress coordinate
            and merge the simulations with the highest progress coordinate.
            If False, split the simulation with the highest progress coordinate
            and merge the simulations with the lowest progress coordinate.
            Default is True.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state. Default is 0.5.
        """
        self.pcoord = pcoord
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.split_low = split_low
        self.target_threshold = target_threshold

    def split(
        self,
        simulations: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Split the simulation with the lowest progress coordinate."""
        # Find the simulations with the lowest progress coordinate
        sorted_indices = np.argsort(self.pcoord)

        # Split the simulations
        if self.split_low:
            indices = sorted_indices[: self.num_resamples].tolist()
        else:
            indices = sorted_indices[-self.num_resamples :].tolist()

        # Split the simulations
        new_sims = self.split_sims(simulations, indices, self.n_split)

        return new_sims

    def merge(
        self,
        simulations: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(self.pcoord)

        # Number of merges is the number of resamples + 1
        # since when we split the simulations we get self.num_resamlpes
        # new simulations and merging them will give us 1 new simulation
        # so we need to merge self.num_resamples + 1 simulations in order
        # to maintain the number of simulations in the ensemble.
        num_merges = self.num_resamples + 1

        # Merge the simulations
        if self.split_low:
            indices = [sorted_indices[-num_merges:].tolist()]
        else:
            indices = [sorted_indices[:num_merges].tolist()]

        # Merge the simulations
        new_sims = self.merge_sims(simulations, indices)

        return new_sims

    def resample(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Resample the weighted ensemble."""
        # Generate the next iteration
        simulations = self.get_next_iteration(current_iteration)

        # Split the simulations
        simulations = self.split(simulations)

        # Merge the simulations
        simulations = self.merge(simulations)

        return simulations

    def recycle(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[int]:
        """Return a list of simulations to recycle."""
        # Recycle the simulations
        if self.split_low:
            indices = [
                i
                for i, p in enumerate(self.pcoord)
                if p < self.target_threshold
            ]
        else:
            indices = [
                i
                for i, p in enumerate(self.pcoord)
                if p > self.target_threshold
            ]

        return indices


def run_inference(
    input_data: list[SimulationResult],
    config: InferenceConfig,
) -> list[SimulationMetadata]:
    """Run inference on the input data."""
    # Extract the pcoord from the last frame of each simulation
    pcoords = [sim_result.pcoord[-1] for sim_result in input_data]

    print(f'Progress coordinates: {pcoords}')
    print(f'Num input simulations: {len(input_data)}')

    # Extract the simulation metadata
    current_iteration = [sim_result.metadata for sim_result in input_data]

    # Resamlpe the ensemble
    resampler = NaiveResampler(
        pcoord=pcoords,
        num_resamples=config.num_resamples,
        n_split=config.n_split,
        split_low=config.split_low,
        target_threshold=config.target_threshold,
    )

    # Get the next iteration of simulations
    next_iteration = resampler.resample(current_iteration)

    return next_iteration
