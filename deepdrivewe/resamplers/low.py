"""Resampler that splits the simulation with the lowest progress coordinate."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from deepdrivewe.api import SimMetadata
from deepdrivewe.resamplers.base import Resampler


class SplitLowResampler(Resampler):
    """Split the simulation with the lowest progress coordinate."""

    def __init__(
        self,
        num_resamples: int = 1,
        n_split: int = 2,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        n_split : int
            The number of simulations to split each simulation into.
            Default is 2.
        pcoord_idx : int
            The index of the progress coordinate to use for splitting and
            merging. Only applicable if a multi-dimensional pcoord is used,
            will choose the specified index of the pcoord for spitting and
            merging. Default is 0.
        """
        super().__init__()
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.pcoord_idx = pcoord_idx

    def split(self, next_sims: list[SimMetadata]) -> list[SimMetadata]:
        """Split the simulation with the lowest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(next_sims, self.pcoord_idx)

        # Find the simulations with the lowest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Split the simulations
        indices = sorted_indices[: self.num_resamples].tolist()

        # Split the simulations
        new_sims = self.split_sims(next_sims, indices, self.n_split)

        return new_sims

    def merge(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(next_sims, self.pcoord_idx)

        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Number of merges is the number of resamples + 1
        # since when we split the simulations we get self.num_resamlpes
        # new simulations and merging them will give us 1 new simulation
        # so we need to merge self.num_resamples + 1 simulations in order
        # to maintain the number of simulations in the ensemble.
        num_merges = self.num_resamples + 1

        # Merge the simulations
        indices = sorted_indices[-num_merges:].tolist()

        # Merge the simulations
        new_sims = self.merge_sims(cur_sims, next_sims, indices)

        return new_sims

    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Resample the weighted ensemble."""
        # Make a copy of the simulations
        cur = deepcopy(cur_sims)
        _next = deepcopy(next_sims)

        # Split the simulations
        _next = self.split(_next)

        # Merge the simulations
        _next = self.merge(cur, _next)

        return cur, _next
