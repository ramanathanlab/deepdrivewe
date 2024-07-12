"""Binning module for WESTPA."""

from __future__ import annotations

import hashlib
import pickle
from abc import ABC
from abc import abstractmethod

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

    def bin_simulations(
        self,
        sims: list[SimMetadata],
    ) -> list[list[SimMetadata]]:
        """Assign the simulations to bins.

        Parameters
        ----------
        sims : list[SimMetadata]
            The simulations to bin.

        Returns
        -------
        list[list[SimMetadata]]
            The binned simulations.
        """
        # Add the binner pickle and hash metadata to the simulations
        for sim in sims:
            sim.binner_pickle, sim.binner_hash = self.pickle_and_hash()

        # Extract the pcoords
        pcoords = np.array([sim.parent_pcoord for sim in sims])

        # Find the bin assignment indices
        assignments = self.assign_bins(pcoords)

        # Set up container for binned sims
        binned_sims: list[list[SimMetadata]] = [[] for _ in range(self.nbins)]

        # Assign the sims to the bins
        for sim, bin_index in zip(sims, assignments):
            binned_sims[bin_index].append(sim)

        # Remove empty bins
        binned_sims = [i for i in binned_sims if i]

        return binned_sims


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
        """Bin the progress coordinate."""
        # Extract the progress coordinates
        pcoord_1d = [pcoord[self.pcoord_idx] for pcoord in pcoords]

        # Bin the progress coordinates (make sure the target state
        # boundary is included in the target state bin).
        return np.digitize(pcoord_1d, self.bins, right=True)
