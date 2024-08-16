"""Implements a two-step resampler utilizing LOF in latent space."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from deepdrivewe.api import SimMetadata
from deepdrivewe.resamplers.base import Resampler


class LOFLowResampler(Resampler):
    """Implements a two-step resampler utilizing LOF in latent space.

    The resampler is designed to be used without bins and follows 2 steps:
        1. Sort the walkers by LOF in latent space and divide the list into
            two groups: the outliers (up for splitting) and inliers (up for
            merging).
        2. Sort the outliers and inliers by pcoord, splitting lowst pcoord
            outliers and highest pcoord inliers.
    """

    def __init__(
        self,
        consider_for_resampling: int = 12,
        max_allowed_weight: float = 0.25,
        min_allowed_weight: float = 10e-40,
        max_resamples: int = 4,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        max_num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        pcoord_idx : int
            The index of the progress coordinate to use for splitting and
            merging. Only applicable if a multi-dimensional pcoord is used,
            will choose the specified index of the pcoord for spitting and
            merging. Default is 0.
        """
        super().__init__()
        self.consider_for_resampling = consider_for_resampling
        self.max_allowed_weight = max_allowed_weight
        self.min_allowed_weight = min_allowed_weight
        self.max_resamples = max_resamples
        self.pcoord_idx = pcoord_idx

    def remove_underweight(
        self,
        sims: list[SimMetadata],
        inds: list[int],
    ) -> list[int]:
        """Remove simulations with weight less than min_allowed_weight."""
        return [i for i in inds if sims[i].weight >= self.min_allowed_weight]

    def remove_overweight(
        self,
        sims: list[SimMetadata],
        inds: list[int],
    ) -> list[int]:
        """Remove simulations with weight greater than max_allowed_weight."""
        return [i for i in inds if sims[i].weight <= self.max_allowed_weight]

    def get_num_resamples(
        self,
        num_split_sims: int,
        num_merge_sims: int,
    ) -> int:
        """Determine the number of resamples to perform."""
        return min(self.max_resamples, num_split_sims, int(num_merge_sims / 2))

    def get_combination(self, tot: int, length: int) -> list[int]:
        """Make all possible combinations of `length` that sums to `tot`.

        Parameters
        ----------
        tot : int
            The total number of segments to sum to (could require
            adding or removing segments).
        length : int
            The number of available segments for resampling.
        """

        # Define a recursive function to generate all possible combinations
        def generate_combinations(
            target: int,
            length: int,
            cur_combo: list[int],
        ) -> list[list[int]]:
            # Base cases
            if length == 0:
                return [cur_combo] if target == 0 else []
            if target <= 0:
                return []

            # Initialize the list of combinations
            combos = []

            # Start the loop from the last number in the current combo
            start = cur_combo[-1] if cur_combo else 1

            # Generate all possible combinations
            for num in range(start, tot + 1):
                # Generate a new combination
                new_combo = [*cur_combo, num]

                # Recursively generate the combinations
                combos.extend(
                    generate_combinations(target - num, length - 1, new_combo),
                )

            return combos

        # Initialize the list of combinations
        combos: list[list[int]] = []

        # Loop over the number of segments
        for segs in range(1, length + 1):
            # Call the recursive function to generate the combinations
            combos.extend(generate_combinations(tot, segs, []))

        # Randomly select one of the combinations
        chosen = combos[np.random.choice(len(combos))]
        # Add 1 to each element in the chosen combination to adjust from
        # the number to add or remove to the number of splits or merges)
        chosen = [i + 1 for i in chosen]

        # Return a sorted list of the chosen combination
        return sorted(chosen, reverse=True)

    def split_with_combination(
        self,
        next_sims: list[SimMetadata],
        inds: list[int],
        num_resamples: int,
    ) -> list[SimMetadata]:
        """Split the outlying simulations with the lowest pcoords."""
        # Get the sims that correspond to the outliers
        outlier_sims = [next_sims[i] for i in inds]

        # Extract the progress coordinates
        pcoords = self.get_pcoords(outlier_sims, self.pcoord_idx)

        # Get the sims with the lowest progress coordinates
        sorted_indices = np.argsort(pcoords)

        # Get the list of number of splits
        num_splits = self.get_combination(num_resamples, len(inds))

        # Get the indices of the sims to split based on the combination
        indices = [inds[i] for i in sorted_indices[: len(num_splits)]]

        # Split the simulations
        new_sims = self.split_sims(next_sims, indices, num_splits)

        return new_sims

    def merge_with_combination(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        inds: list[int],
        num_resamples: int,
    ) -> list[SimMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Get the sims that correspond to the inliers
        inlier_sims = [next_sims[i] for i in inds]

        # Extract the progress coordinates
        pcoords = self.get_pcoords(inlier_sims, self.pcoord_idx)

        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Get the list of number of merges
        num_merges = self.get_combination(num_resamples, len(inds))

        # Loop over the number of merges
        for merge in range(len(num_merges)):
            # Get the indices of the sims to merge based on the combination
            indices = [inds[i] for i in sorted_indices[-num_merges[merge] :]]

            # Remove the used indices from the end of the sorted indices
            sorted_indices = sorted_indices[: -num_merges[merge]]

            # Merge the simulations
            next_sims = self.merge_sims(cur_sims, next_sims, indices)

        return next_sims

    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Resample the weighted ensemble."""
        # Make a copy of the simulations
        cur = deepcopy(cur_sims)
        _next = deepcopy(next_sims)

        # Get the LOFs; assumes LOF is saved as a second parent pcoord in
        # metadata
        lof = self.get_pcoords(_next, 1)

        # Get the indices to sort the simulations
        sorted_indices = np.argsort(lof)

        # Get the indices of the outliers and inliers
        outliers = sorted_indices[: self.consider_for_resampling].tolist()
        inliers = sorted_indices[self.consider_for_resampling :].tolist()

        # Remove underweight and overweight sims
        outliers = self.remove_underweight(_next, outliers)
        inliers = self.remove_overweight(_next, inliers)

        # Determine the number of resamples to perform
        num_resamples = self.get_num_resamples(len(outliers), len(inliers))

        # If there are enough walkers in the thresholds, split and merge
        if num_resamples > 0:
            # Find the inlier sims before the list is modified
            inlier_ids = [
                sim.simulation_id
                for i, sim in enumerate(_next)
                if i in inliers
            ]

            # Split the simulations
            _next = self.split_with_combination(_next, outliers, num_resamples)

            # Find the indices that correspond to the inliers
            # NOTE: This is pretty hokey but is necessary because the list of
            # simulations is modified in the split step
            inliers = [
                i
                for i, sim in enumerate(_next)
                if sim.simulation_id in inlier_ids
            ]

            # Merge the simulations
            _next = self.merge_with_combination(
                cur,
                _next,
                inliers,
                num_resamples,
            )

        return cur, _next
