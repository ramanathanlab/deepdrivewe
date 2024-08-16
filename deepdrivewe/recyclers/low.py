"""Recycle simulations under a certain progress coordinate threshold."""

from __future__ import annotations

import numpy as np

from deepdrivewe.api import BasisStates
from deepdrivewe.recyclers.base import Recycler


class LowRecycler(Recycler):
    """Recycle simulations under a certain progress coordinate threshold."""

    def __init__(
        self,
        basis_states: BasisStates,
        target_threshold: float,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the recycler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state.
        pcoord_idx : int
            The index of the progress coordinate to use for recycling. Only
            applicable if a multi-dimensional pcoord is used, will choose the
            specified index of the pcoord for recycling. Default is 0.
        """
        super().__init__(basis_states)
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def recycle(self, pcoords: np.ndarray) -> np.ndarray:
        """Recycle the simulations under the target threshold.

        Parameters
        ----------
        pcoords : np.ndarray
            The progress coordinates for the simulations.
            Shape: (n_simulations, n_dims).

        Returns
        -------
        np.ndarray
            The list of simulation indices to recycle. Shape: (n_recycled,)
        """
        return np.where(pcoords[:, self.pcoord_idx] < self.target_threshold)[0]
