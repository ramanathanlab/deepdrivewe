"""Rectilinear binner."""

from __future__ import annotations

import numpy as np

from deepdrivewe.binners.base import Binner


class RectilinearBinner(Binner):
    """Rectilinear binner for the progress coordinate."""

    def __init__(
        self,
        bins: list[float],
        bin_target_counts: int | list[int],
        target_state_inds: int | list[int] | None = None,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the binner.

        Parameters
        ----------
        bins : list[float]
            The bin edges for the progress coordinate.
        bin_target_counts : int | list[int]
            The target counts for each bin. If an integer is provided,
            the target counts are assumed to be the same for each bin.
        target_state_inds : int | list[int] | None
            The index of the target state. If an integer is provided, then
            there is only one target state. If a list of integers is provided,
            then there are multiple target states. If None is provided, then
            there are no target states. Default is None.
        pcoord_idx : int
            The index of the progress coordinate to use for binning.
            Default is 0.
        """
        super().__init__(bin_target_counts, target_state_inds)

        self.bins = bins
        self.pcoord_idx = pcoord_idx

        # Check that the bins are sorted
        if not np.all(np.diff(self.bins) > 0):
            raise ValueError('Bins must be sorted in ascending order.')

    @property
    def nbins(self) -> int:
        """The number of bins."""
        return len(self.bins) - 1

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
        return np.digitize(pcoords[:, self.pcoord_idx], self.bins, right=True)
