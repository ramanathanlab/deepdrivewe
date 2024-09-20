"""Huber and Kim resampling."""

from __future__ import annotations

from copy import deepcopy

from deepdrivewe.api import SimMetadata
from deepdrivewe.resamplers.base import Resampler


class HuberKimResampler(Resampler):
    """
    Run Huber and Kim resampling.

    The resampling procedure is mostly outlined in Huber, Kim 1996:
        https://doi.org/10.1016/S0006-3495(96)79552-8

    with the additions of the adjust counts and weight thresholds that make
    this more closely replicate the base implementation of fixed bins from
    WESTPA: http://dx.doi.org/10.1021/ct5010615

    This resampler is designed to be used with bins and follows 3 steps:
        1. Resample based on weight
        2. Adjust the number of simulations in each bin
        3. Split and merge to keep simulations inside weight thresholds
    """

    def __init__(
        self,
        sims_per_bin: int = 5,
        max_allowed_weight: float = 1.0,
        min_allowed_weight: float = 10e-40,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        sims_per_bin : int
            The number of simulations to have in each bin. Default is 5.
        max_allowed_weight : float
            The maximum allowed weight for each simulation. If the weight of a
            simulation exceeds this value, it will be split. Default is 1.0.
        min_allowed_weight : float
            The minimum allowed weight for each simulation. All the simulations
            with a weight less than this value will be merged into a single
            simulation walker. Default is 10e-40.
        """
        super().__init__()
        self.sims_per_bin = sims_per_bin
        self.max_allowed_weight = max_allowed_weight
        self.min_allowed_weight = min_allowed_weight

    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Resample the weighted ensemble."""
        # Make a copy of the simulations
        cur = deepcopy(cur_sims)
        _next = deepcopy(next_sims)

        # Get the weight of the simulations
        weights = [sim.weight for sim in _next]

        # Calculate the ideal weight
        ideal_weight = sum(weights) / self.sims_per_bin

        # Split the simulations by weight
        _next = self.split_by_weight(_next, ideal_weight)

        # Merge the simulations by weight
        _next = self.merge_by_weight(cur, _next, ideal_weight)

        # Adjust the number of simulations in each bin
        _next = self.adjust_count(cur, _next, self.sims_per_bin)

        # Split the simulations by threshold
        _next = self.split_by_threshold(_next, self.max_allowed_weight)

        # Merge the simulations by threshold
        _next = self.merge_by_threshold(cur, _next, self.min_allowed_weight)

        return cur, _next
