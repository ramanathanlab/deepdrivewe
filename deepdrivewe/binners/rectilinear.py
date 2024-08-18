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
        target_state_inds: int | list[int] = 0,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the binner.

        Parameters
        ----------
        bins : list[float]
            The bin edges for the progress coordinate.
        pcoord_idx : int
            The index of the progress coordinate to use for binning.
            Default is 0.
        bin_target_counts : int | list[int]
            The target counts for each bin. If an integer is provided,
            the target counts are assumed to be the same for each bin.
        target_state_inds : int | list[int]
            The index of the target state. If an integer is provided, then
            there is only one target state. If a list of integers is provided,
            then there are multiple target states. Default is 0 which
            corresponds to the first bin.
        """
        self.bins = bins
        self.bin_target_counts = bin_target_counts
        self.target_state_inds = target_state_inds
        self.pcoord_idx = pcoord_idx

        # Check that the bins are sorted
        if not np.all(np.diff(self.bins) > 0):
            raise ValueError('Bins must be sorted in ascending order.')

    @property
    def nbins(self) -> int:
        """The number of bins."""
        return len(self.bins) - 1

    def get_bin_target_counts(self) -> list[int]:
        """Get the target counts for each bin.

        Returns
        -------
        list[int]
            The target counts for each bin.
        """
        # Check if the bin target counts is an integer
        # If so, then set the target counts for each bin to the same value
        # and set the target state bins to 0. Cache the result.
        if isinstance(self.bin_target_counts, int):
            # Create a list of the bin target counts
            bin_target_counts = [self.bin_target_counts] * self.nbins

            # Get the target state indices (convert to a list if an integer)
            if isinstance(self.target_state_inds, int):
                self.target_state_inds = [self.target_state_inds]

            # Set each of the target state bins to 0 since they are recycled
            for i in self.target_state_inds:
                bin_target_counts[i] = 0

            # Cache the result
            self.bin_target_counts = bin_target_counts

        # Otherwise, return the list of bin target counts
        return self.bin_target_counts

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
