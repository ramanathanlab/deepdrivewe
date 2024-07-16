"""Binning module for WESTPA."""

from __future__ import annotations

import hashlib
import pickle
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy

import numpy as np

from westpa_colmena.ensemble import SimMetadata


class Binner(ABC):
    """Binner for the progress coordinate."""

    @property
    @abstractmethod
    def nbins(self) -> int:
        """The number of bins."""
        ...

    @abstractmethod
    def assign_bins(self, pcoords: np.ndarray) -> np.ndarray:
        """Assign the simulation pcoords to bins."""
        ...

    @property
    def labels(self) -> list[str]:
        """The bin labels for WESTPA."""
        return [f'state{i}' for i in range(self.nbins)]

    def assign(
        self,
        coords: np.ndarray,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Assign the simulations to bins.

        This API is compatible with the WESTPA Binner class.

        Parameters
        ----------
        coords : np.ndarray
            The progress coordinates to bin. Shape: (n_simulations, n_dims).
        mask : np.ndarray
            The mask to apply to skip a certain simulation (0 skips and 1
            uses the simulation). By default all simulations are used.
            Shape: (n_simulations,)
        output : np.ndarray
            The output array to store the bin assignments.
            Shape: (n_simulations,)

        Returns
        -------
        np.ndarray
            The bin assignments for each simulation (n_simulations,)
        """
        # Note: The mask is not used in this implementation (i.e., all
        # simulations are used).

        # Assign the simulations to bin indices
        output = self.assign_bins(coords)

        return output

    def pickle_and_hash(self) -> tuple[bytes, str]:
        """Pickle this mapper and calculate a hash of the result.

        Pickle this mapper and calculate a hash of the result
        (thus identifying the contents of the pickled data), returning a
        tuple ``(pickled_data, hash)``. This will raise PickleError if this
        mapper cannot be pickled, in which case code that would otherwise
        rely on detecting a topology change must assume a topology change
        happened, even if one did not.
        """
        pkldat = pickle.dumps(self, pickle.HIGHEST_PROTOCOL)
        binhash = hashlib.sha256(pkldat)
        return (pkldat, binhash.hexdigest())

    def _get_bin_stats(
        self,
        bin_assignments: dict[int, list[int]],
        cur_sims: list[SimMetadata],
    ) -> tuple[float, float]:
        """Compute the bin statistics.

        Parameters
        ----------
        bin_assignments : dict[int, list[int]]
            A dictionary of the bin assignments. The keys are the bin
            indices and the values are the indices of the simulations
            assigned to that bin.

        cur_sims : list[SimMetadata]
            The list of current simulations.

        Returns
        -------
        tuple[float, float]
            The minimum and maximum bin probabilities.
        """
        # Compute the probability of each bin by summing the weights
        bin_probs = []

        # Iterate over the bin assignments
        for sim_indices in bin_assignments.values():
            # Extract the simulations in the bin
            binned_sims = [cur_sims[i] for i in sim_indices]

            # Compute the probability of the bin
            bin_prob = sum(x.weight for x in binned_sims)

            # Append the bin probability
            bin_probs.append(bin_prob)

        # Compute the min and max bin probabilities
        min_bin_prob = min(bin_probs)
        max_bin_prob = max(bin_probs)

        return min_bin_prob, max_bin_prob

    def bin_simulations(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[dict[int, list[int]], list[SimMetadata]]:
        """Assign the simulations to bins.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
            The list of next simulations.

        Returns
        -------
        dict[int, list[int]]
            A dictionary of the bin assignments. The keys are the bin
            indices and the values are the indices of the simulations
            assigned to that bin.

        list[SimMetadata]
            The updated current simulations with metadata added.
        """
        # Make a deep copy of the simulations to prevent side effects
        _cur_sims = deepcopy(cur_sims)

        # Extract the pcoords using the parent pcoords since
        # they are they have already been recycled.
        pcoords = np.array([sim.parent_pcoord for sim in next_sims])

        # Find the bin assignment indices
        assignments = self.assign_bins(pcoords)

        # Collect a dictionary of the bin assignments
        bin_assignments = defaultdict(list)

        # Check that the number of assignments is the same as the simulations
        assert len(assignments) == len(next_sims)

        # Assign the simulations to the bins
        for sim_idx, bin_idx in enumerate(assignments):
            bin_assignments[bin_idx].append(sim_idx)

        # Update the current simulation metadata
        for sim in _cur_sims:
            # Add the binner pickle and hash metadata to the simulations
            sim.binner_pickle, sim.binner_hash = self.pickle_and_hash()

            # Add the bin statistics to the simulations
            sim.min_bin_prob, sim.max_bin_prob = self._get_bin_stats(
                bin_assignments,
                _cur_sims,
            )

        return bin_assignments, _cur_sims


class RectilinearBinner(Binner):
    """Rectilinear binner for the progress coordinate."""

    def __init__(self, bins: list[float], pcoord_idx: int = 0) -> None:
        """Initialize the binner.

        Parameters
        ----------
        bins : list[float]
            The bin edges for the progress coordinate.
        pcoord_idx : int
            The index of the progress coordinate to use for binning.
            Default is 0.
        """
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
