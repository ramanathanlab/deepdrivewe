"""Implements a two-step resampler utilizing LOF in latent space."""

from __future__ import annotations

import random
from copy import deepcopy
from itertools import combinations_with_replacement

import pandas as pd

from deepdrivewe.api import SimMetadata
from deepdrivewe.resamplers.base import Resampler


class LOFLowResamplerV2(Resampler):
    """Implements a two-step resampler utilizing LOF in latent space.

    The resampler is designed to be used without bins and follows 2 steps:
        1.  Sort the walkers by LOF in latent space and divide the list into
            two groups: the outliers (up for splitting) and inliers (up for
            merging). `consider_for_resampling` determines the number of sims
            in each group to consider for resampling (the rest are left alone).
        2.  Sort the outliers and inliers by pcoord, splitting lowest pcoord
            outliers and merging highest pcoord inliers.
    """

    def __init__(
        self,
        consider_for_resampling: int = 12,
        max_resamples: int = 4,
        max_allowed_weight: float = 0.1,
        min_allowed_weight: float = 10e-40,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        consider_for_resampling : int
            The number of simulations to consider for resampling.
            Default is 12.
        max_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 4.
        max_allowed_weight : float
            The maximum allowed weight for a simulation. Default is 0.1.
        min_allowed_weight : float
            The minimum allowed weight for a simulation. Default is 10e-40.
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

    def _get_combination(self, total: int, length: int) -> list[int]:
        def generate_combinations(n: int, max_length: int) -> list[list[int]]:
            if n == 0:
                return [[]]
            if max_length == 0:
                return []

            combinations = []
            for i in range(1, n + 1):
                for tail in generate_combinations(n - i, max_length - 1):
                    combinations.append([i, *tail])
            return combinations

        # Filter out combinations that don't match the required length
        combs = [
            combo
            for combo in generate_combinations(total, length)
            if len(combo) <= length
        ]

        print(f'[_get_combination] {combs=}')

        # Only keep the unique combinations
        unique_combs = {tuple(sorted(combo)) for combo in combs}

        print(f'[_get_combination] {unique_combs=}')

        # Randomly select one of the combinations
        choice = random.choice(list(unique_combs))

        print(f'[_get_combination] {choice=}')

        # Add 1 to each element in the chosen combination to adjust from
        # the number to add or remove to the number of splits or merges
        chosen = [i + 1 for i in choice]

        print('[_get_combination] chosen + 1', chosen, flush=True)

        return chosen

    def _get_combination_old_v(self, total: int, length: int) -> list[int]:
        """Get the number of splits or merges to perform for each simulation.

        Parameters
        ----------
        total : int
            The total number of simulations to sum to (could require
            adding or removing simulations).
        length : int
            The number of available simulations for resampling.

        Returns
        -------
        list[int]
            The number of splits or merges to perform for each simulation.
        """
        # Generate all combinations of numbers from 1 to length - 1
        combinations = combinations_with_replacement(range(1, length), total)

        # Filter the combinations to only include those that sum to total
        combs = [list(comb) for comb in combinations if sum(comb) == total]

        print(f'[_get_combination] {combs=}')

        # Randomly select one of the combinations
        chosen = random.choice(combs)

        print(f'[_get_combination] {chosen=}')

        # Add 1 to each element in the chosen combination to adjust from
        # the number to add or remove to the number of splits or merges
        chosen = [i + 1 for i in chosen]

        print('[_get_combination] chosen + 1', chosen, flush=True)

        return chosen

    def split_with_combination(
        self,
        outliers: pd.DataFrame,
        next_sims: list[SimMetadata],
        num_resamples: int,
    ) -> list[SimMetadata]:
        """Split the outlying simulations with the lowest pcoords."""
        # Compute a random combination of splits to perform
        n_splits = self._get_combination(num_resamples, len(outliers))

        # print(f'[split_with_combination] {n_splits=}')

        # Sort the number of splits in descending order so that the
        # smallest RMSD simulations are split the most.
        n_splits = sorted(n_splits, reverse=True)

        # print(f'[split_with_combination] reverse sorted {n_splits=}')

        # Get the indices of the N simulations with lowest RMSD values
        indices = (
            outliers.sort_values('rmsd')  # Sort by RMSD
            .head(len(n_splits))  # Take the N lowest RMSD values
            .index  # Get the indices
        )

        # print(f'[split_with_combination] {outliers=}')
        # print(f'[split_with_combination] {outliers.sort_values("rmsd")=}')
        # print(f'[split_with_combination] {indices=}', flush=True)

        # Split the simulations
        new_sims = self.split_sims(next_sims, indices.tolist(), n_splits)

        return new_sims

    def merge_with_combination(
        self,
        inliers: pd.DataFrame,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        num_resamples: int,
    ) -> list[SimMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Get the list of number of merges
        n_merges = self._get_combination(num_resamples, len(inliers))

        # print(f'[merge_with_combination] {n_merges=}')

        # Sort the number of merges in ascending order so that the
        # largest RMSD simulations are merged the most.
        n_merges = sorted(n_merges)

        # print(f'[merge_with_combination] sorted {n_merges=}')

        # Loop over the number of merges
        for merge in n_merges:
            # Get the indices of the sims to merge based on the combination
            indices = inliers.sort_values('rmsd').tail(merge).index

            # print(f'[merge_with_combination] {inliers=}')
            # print(f'[merge_with_combination] {indices=}', flush=True)

            # Remove the used indices from the dataframe
            inliers = inliers.drop(indices)

            # print(
            #     f'[merge_with_combination] after drop {inliers=}',
            #     flush=True,
            # )

            # Merge the simulations
            next_sims = self.merge_sims(cur_sims, next_sims, indices.tolist())

        return next_sims

    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Resample the weighted ensemble.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The current simulations.
        next_sims : list[SimMetadata]
            The next simulations.

        Returns
        -------
        tuple[list[SimMetadata], list[SimMetadata]]
            The resampled current and next simulations.

        Raises
        ------
        ValueError
            If consider_for_resampling is too large for the number of sims.
        """
        # Check if there are enough sims to resample
        if len(next_sims) < 2 * self.consider_for_resampling:
            raise ValueError(
                f'consider_for_resampling={self.consider_for_resampling} '
                f'is too large for the number of sims {len(next_sims)}.',
                'Consider increasing the number of simulations or decreasing',
                'consider_for_resampling such that it is less than or equal '
                'to half of the number of simulations.',
            )

        # Make a copy of the simulations
        cur = deepcopy(cur_sims)
        _next = deepcopy(next_sims)

        # Get the RMSD values which assumes are saved as the first pcoord
        rmsd = self.get_pcoords(_next, pcoord_idx=0)

        # First get the LOF scores which assumes are saved as the second
        # (parent) progress coordinate in the next simulations.
        lof = self.get_pcoords(_next, pcoord_idx=1)

        df = (
            pd.DataFrame(
                {
                    'rmsd': rmsd,
                    'lof': lof,
                    'weight': [sim.weight for sim in _next],
                },
            ).sort_values('lof')  # First sort by lof (small are outliers)
        )

        print(df, flush=True)

        # Take the smallest num_outliers lof scores
        outliers = df.head(self.consider_for_resampling)

        # Take the largest num_inliers lof scores
        inliers = df.tail(self.consider_for_resampling)

        # Remove underweight simulations from the outliers so we don't split
        # simulations that are too small. Note that it's fine to keep small
        # weight simulations in the inliers because they will be merged and
        # the weights will be increased.
        outliers = outliers[outliers.weight >= self.min_allowed_weight]

        # Remove overweight simulations from the inliers so we don't merge
        # simulations that are too large. Note that it's fine to keep large
        # weight simulations in the outliers because they will be split and
        # the weights will be reduced.
        inliers = inliers[inliers.weight <= self.max_allowed_weight]

        # Determine the number of resamples to perform. If removing the
        # underweight or overweight sims results in not enough simulations
        # to split or merge, then dynamically adjust the number of resamples
        # to allow for the maximum number of splits and merges possible. This
        # helps to balance the weights of the simulations by preventing very
        # aggressive split (e.g., splitting a single sim to many) or merges
        # (e.g., merging many sims to a single sim) which helps to prevent
        # the weights from becoming too large (biased) or too small (not
        # meaningful for rate constant estimation).
        num_resamples = min(
            self.max_resamples,  # The user defined maximum number of resamples
            len(outliers),  # Roughly the number of possible splits
            int(len(inliers) / 2),  # Roughly the number of possible merges
        )

        print(f'Resampling with {num_resamples=}', flush=True)

        # If there are enough walkers in the thresholds, split and merge
        if num_resamples > 0:
            # Split the simulations
            _next = self.split_with_combination(outliers, _next, num_resamples)

            # Merge the simulations
            _next = self.merge_with_combination(
                inliers,
                cur,
                _next,
                num_resamples,
            )

        return cur, _next
