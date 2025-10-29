"""Multirectilinear binner."""

from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic_2d

from deepdrivewe.binners.base import Binner


class MultiRectilinearBinner(Binner):
    """Multirectilinear binner for multiple progress coordinates."""

    def __init__(
        self,
        bins: list[np.ndarray | list[float]],
        bin_target_counts: int | list[int],
        target_state_inds: int | list[int] | None = None,
    ) -> None:
        """Initialize the binner.

        Parameters
        ----------
        bins : list[np.ndarray | list[float]]
            The bin edges for the progress coordinates.
        bin_target_counts : int | list[int]
            The target counts for each bin. If an integer is provided,
            the target counts are assumed to be the same for each bin.
        target_state_inds : int | list[int] | None
            The index of the target state. If an integer is provided, then
            there is only one target state. If a list of integers is provided,
            then there are multiple target states. If None is provided, then
            there are no target states. Default is None.
        """
        super().__init__(bin_target_counts, target_state_inds)

        self.bins = bins

        # Check that the bins are sorted
        for binbounds in self.bins:
            if not np.all(np.diff(binbounds) > 0):
                raise ValueError(
                    'Bin boundaries must be sorted in ascending order.',
                )

    @property
    def nbins(self) -> int:
        """The number of bins."""
        # Calculate the number of bins per dimension
        nbins_per_dim = np.array([len(dim) - 1 for dim in self.bins])

        # Calculate the total number of bins
        return int(np.prod(nbins_per_dim))

    def assign_bins(self, pcoords: np.ndarray) -> np.ndarray:
        """Bin the progress coordinate.

        Parameters
        ----------
        pcoords : np.ndarray
            The progress coordinates to bin. Shape: (n_simulations, n_dims).

        Returns
        -------
        np.ndarray
            The bin assignments for each simulation. Shape: (n_simulations,)
        """
        # Bin the progress coordinates (make sure the target state
        # boundary is included in the target state bin).
        _, x_edge, _, bid = binned_statistic_2d(
            *pcoords.T,
            values=None,
            statistic='count',
            bins=self.bins,
            expand_binnumbers=True,
        )

        # Calculate the bin indices in row-major order
        bin_ids = np.array(
            [(ibid[0] - 1) * (len(x_edge) - 1) + ibid[1] for ibid in bid],
        )

        # Check that the number of bin indices is the same as the
        # number of simulations
        if len(bin_ids) != len(pcoords):
            raise ValueError(
                'Number of bin indices must match the number of simulations.',
            )

        return bin_ids
