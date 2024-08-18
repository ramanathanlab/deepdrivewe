"""Binning module for WESTPA."""

from __future__ import annotations

import hashlib
import pickle
from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import numpy as np

from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata


class Binner(ABC):
    """Binner for the progress coordinate."""

    @abstractmethod
    def get_bin_target_counts(self) -> list[int]:
        """Get the target counts for each bin.

        Returns
        -------
        list[int]
            The target counts for each bin.
        """
        ...

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
        # Initialize output if not provided
        if output is None:
            output = np.empty(coords.shape[0], dtype=np.uint16)

        # Initialize the mask if not provided
        if mask is not None:
            mask = np.ones(coords.shape[0], dtype=np.bool_)

        # Assign the simulations to bin indices (in-place)
        output[mask] = self.assign_bins(coords)

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

    def _get_bin_assignments(
        self,
        pcoords: np.ndarray,
    ) -> dict[int, list[int]]:
        # Find the bin assignment indices
        assignments = self.assign_bins(pcoords)

        # Check that the number of assignments is the same as the simulations
        if len(assignments) != len(pcoords):
            raise ValueError(
                'Number of assignments must match the number of simulations.',
            )

        # Collect a dictionary of the bin assignments
        bin_assignments = defaultdict(list)

        # Assign the simulations to the bins
        for sim_idx, bin_idx in enumerate(assignments):
            bin_assignments[bin_idx].append(sim_idx)

        return bin_assignments

    def _get_bin_probs(
        self,
        bin_assignments: dict[int, list[int]],
        cur_sims: list[SimMetadata],
    ) -> list[float]:
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
        list[float]
            The sum of weights in each bin (i.e., bin probabilities).
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

        return bin_probs

    def compute_iteration_metadata(
        self,
        cur_sims: list[SimMetadata],
    ) -> IterationMetadata:
        """Compute the iteration metadata using the current simulations.

        Returns
        -------
        IterationMetadata
            The iteration metadata.
        """
        # Extract the pcoords from the last frame of each simulation
        pcoords = np.array([sim.pcoord[-1] for sim in cur_sims])

        # Assign the simulations to bins
        bin_assignments = self._get_bin_assignments(pcoords)

        # Compute the bin probabilities
        bin_probs = self._get_bin_probs(bin_assignments, cur_sims)

        # Add the binner pickle and hash metadata to the iteration
        binner_pickle, binner_hash = self.pickle_and_hash()

        # Create the iteration metadata
        return IterationMetadata(
            iteration_id=cur_sims[0].iteration_id,
            binner_pickle=binner_pickle,
            binner_hash=binner_hash,
            min_bin_prob=min(bin_probs),
            max_bin_prob=max(bin_probs),
            bin_target_counts=self.get_bin_target_counts(),
        )

    def bin_simulations(
        self,
        next_sims: list[SimMetadata],
    ) -> dict[int, list[int]]:
        """Assign the simulations to bins.

        Parameters
        ----------
        next_sims : list[SimMetadata]
            The list of next simulations.

        Returns
        -------
        dict[int, list[int]]
            A dictionary of the bin assignments. The keys are the bin
            indices and the values are the indices of the simulations
            assigned to that bin.
        """
        # Extract the pcoords using the parent pcoords since
        # they are they have already been recycled.
        pcoords = np.array([sim.parent_pcoord for sim in next_sims])

        # Assign the simulations to bins
        bin_assignments = self._get_bin_assignments(pcoords)

        return bin_assignments
