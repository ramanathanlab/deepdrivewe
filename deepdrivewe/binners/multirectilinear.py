"""Multirectilinear binner."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import binned_statistic_dd

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
        _, bin_edges, bid = binned_statistic_dd(
            np.asarray(pcoords),
            values=None,
            statistic='count',
            bins=self.bins,
            expand_binnumbers=True,
        )

        # Clip the bin indices so any index outside of defined bins are moved
        # to nearest defined bin
        nbins_per_dim = [len(edges) - 1 for edges in bin_edges]

        # If binning a 1D coordinate, a 1D array will be returned.
        bid = np.atleast_2d(bid)

        for idx, ibid in enumerate(bid):
            if not np.all(ibid > 0) or not np.all(ibid < len(self.bins[idx])):
                warnings.warn(
                    'Simulations with progress coordinates outside the bin '
                    f'boundaries definition of dimension {idx} are '
                    'automatically placed into the nearest terminal bins. '
                    'Consider modifying your bin boundaries by adding '
                    "'np.inf' or '-np.inf' on either end of your bin "
                    'definitions.',
                    stacklevel=2,
                )
                bid[idx] = np.clip(ibid, 1, nbins_per_dim[idx])

        # Calculate the bin indices in row-major order
        bin_ids = np.zeros(len(pcoords), dtype=int)
        for idx, ibid in enumerate(bid.T):
            for idim in range(len(nbins_per_dim) - 1):
                bin_ids[idx] += (ibid[idim] - 1) * nbins_per_dim[idim]
            bin_ids[idx] += ibid[-1] - 1

        # Check that the number of bin indices is the same as the
        # number of simulations
        if len(bin_ids) != len(pcoords):
            raise ValueError(
                'Number of bin indices must match the number of simulations.',
            )

        return bin_ids
