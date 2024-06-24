"""Resampling algorithms for the weighted ensemble."""

from __future__ import annotations

import itertools
import math
from abc import ABC
from abc import abstractmethod

import numpy as np

from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import SimMetadata


class Resampler(ABC):
    """Resampler for the weighted ensemble."""

    def __init__(self, basis_states: BasisStates) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        """
        self.basis_states = basis_states

        # Create a counter to keep track of the simulation IDs
        self.index_counter = itertools.count()

    def get_next_iteration(
        self,
        current_iteration: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Return the simulations for the next iteration."""
        # Reset the index counter
        self.index_counter = itertools.count()

        # Get the recycled simulation indices
        recycle_indices = self.recycle(current_iteration)

        # Create a list to store the new simulations for this iteration
        simulations = []

        for idx, sim in enumerate(current_iteration):
            # Ensure that the simulation has a restart file, i.e., the `sim`
            # object represents a simulation that has been run.
            assert sim.restart_file is not None

            # Check if the simulation should be recycled
            if idx in recycle_indices:
                # Choose a random basis state to restart the simulation from
                basis_idx = np.random.choice(len(self.basis_states))
                basis_state = self.basis_states[basis_idx]
                # Set the parent restart file to the basis state
                parent_restart_file = basis_state.parent_restart_file
                # Set the prev simulation ID to the negative of previous
                # simulation to indicate that the simulation is recycled
                parent_simulation_id = -1 * sim.simulation_id
                parent_pcoord = basis_state.parent_pcoord
            else:
                # If the simulation is not recycled, set the parent restart
                # file and simulation id to the restart file of the current
                # simulation
                assert sim.restart_file is not None
                assert sim.pcoord is not None
                parent_restart_file = sim.restart_file
                parent_simulation_id = sim.simulation_id
                parent_pcoord = sim.pcoord

            # Create the metadata for the new simulation
            new_sim = SimMetadata(
                weight=sim.weight,
                simulation_id=idx,
                iteration_id=sim.iteration_id + 1,
                parent_simulation_id=parent_simulation_id,
                parent_restart_file=parent_restart_file,
                parent_pcoord=parent_pcoord,
            )

            # Add the new simulation to the current iteration
            simulations.append(new_sim)

        return simulations

    def _add_new_simulation(
        self,
        sim: SimMetadata,
        weight: float,
    ) -> SimMetadata:
        """Add a new simulation to the current iteration."""
        # Create the metadata for the new simulation
        return SimMetadata(
            weight=weight,
            simulation_id=next(self.index_counter),
            iteration_id=sim.iteration_id,
            parent_simulation_id=sim.parent_simulation_id,
            restart_file=sim.restart_file,
            parent_restart_file=sim.parent_restart_file,
            parent_pcoord=sim.parent_pcoord,
        )

    def split_sims(
        self,
        sims: list[SimMetadata],
        indices: list[int],
        n_splits: int | list[int] = 2,
    ) -> list[SimMetadata]:
        """Split the simulation index into `n_split`."""
        # Get the simulations to split
        sims_to_split = [sims[idx] for idx in indices]

        # Handle the case where `n_split` is a single integer
        if isinstance(n_splits, int):
            n_splits = [n_splits] * len(sims_to_split)

        # Create a list to store the new simulations
        new_sims: list[SimMetadata] = []

        # Add back the simulations that will not be split
        new_sims.extend(sims[i] for i in range(len(sims)) if i not in indices)

        # Split the simulations using the specified number of splits
        # and equal weights for the split simulations
        for sim, n_split in zip(sims_to_split, n_splits):
            for _ in range(n_split):
                new_sim = self._add_new_simulation(sim, sim.weight / n_split)
                new_sims.append(new_sim)

        return new_sims

    def merge_sims(
        self,
        sims: list[SimMetadata],
        indices: list[list[int]],
    ) -> list[SimMetadata]:
        """Merge each group of simulation indices into a single simulation."""
        # Get the indices of non-merged simulations
        print(f'{indices=}', flush=True)
        merge_idxs = [idx for index_group in indices for idx in index_group]
        no_merge_idxs = [i for i in range(len(sims)) if i not in merge_idxs]

        # Create a list to store the new simulations
        new_sims: list[SimMetadata] = []

        # Add back the simulations that will not be merged
        new_sims.extend(sims[i] for i in no_merge_idxs)

        for index_group in indices:
            # Get the simulations to merge
            to_merge = [sims[idx] for idx in index_group]

            # Get the weights of each simulation to merge
            weights = [sim.weight for sim in to_merge]

            # Make sure the weights are normalized to sum to 1 for randomizing.
            # Since the entire ensemble should have a total weight of 1
            # any subset of the ensemble will have a total weight less than 1.
            norm_weights = np.array(weights) / sum(weights)

            # Randomly select one of the simulations with probability equal
            # to the normalized weights
            select: int = np.random.choice(len(to_merge), p=norm_weights)

            # Add the new simulation to the current iteration
            new_sim = self._add_new_simulation(to_merge[select], sum(weights))

            # Add the new simulation to the list of new simulations
            new_sims.append(new_sim)

        # Return the new simulation
        return new_sims

    def split_by_weight(
        self,
        sims: list[SimMetadata],
        ideal_weight: float,
    ) -> list[SimMetadata]:
        """Split overweight sims.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations in a particular bin to split.
        ideal_weight : float
            The ideal weight for each simulation, defined as the total (sum)
            weight of bin divided by the desired number of walkers in the bin.
            This is roughly equivalent to the average weight of the simulations
            in the bin.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after splitting.
        """
        # Get the weights of the simulations
        weights = np.array([sim.weight for sim in sims])

        # Get the simulation indices
        indices = np.arange(len(sims))

        # Find the walkers that need to be split
        split_inds = indices[weights > ideal_weight].tolist()

        # Calculate the number of splits for each walker
        num_splits = [math.ceil(weights[i] / ideal_weight) for i in split_inds]

        # Split the simulations
        return self.split_sims(sims, split_inds, num_splits)

    def merge_by_weight(
        self,
        sims: list[SimMetadata],
        ideal_weight: float,
    ) -> list[SimMetadata]:
        """Merge underweight sims.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations in a particular bin to merge.
        ideal_weight : float
            The ideal weight for each simulation, defined as the total (sum)
            weight of bin divided by the desired number of walkers in the bin.
            This is roughly equivalent to the average weight of the simulations
            in the bin.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after merging.
        """
        while True:
            # Sort the simulations by weight
            sorted_sims = sorted(sims, key=lambda sim: sim.weight)

            # Get the weights of the sorted simulations
            weights = np.array([sim.weight for sim in sorted_sims])

            # Accumulate the weights
            cumul_weight = np.add.accumulate(weights)

            # Get the simulation indices
            indices = np.arange(len(sims))

            # Find the walkers that need to be merged
            to_merge = indices[cumul_weight <= ideal_weight].tolist()

            # Break the loop if no walkers need to be merged
            if len(to_merge) < 2:  # noqa: PLR2004
                return sims

            # Merge the simulations
            sims = self.merge_sims(sorted_sims, [to_merge])

    def adjust_count(
        self,
        sims: list[SimMetadata],
        target_count: int,
    ) -> list[SimMetadata]:
        """Adjust the number of sims to match the target count.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations in a particular bin to adjust.
        target_count : int
            The number of simulations to have in the bin.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after adjusting.
        """
        # Case 1: Too few sims
        while len(sims) < target_count:
            # Get the index of the largest weight simulation
            index = int(np.argmax([sim.weight for sim in sims]))

            # Split the highest weight sim in two
            sims = self.split_sims(sims, [index], 2)

            # Break the loop if the target count is reached
            if len(sims) == target_count:
                break

        # Case 2: Too many sims
        while len(sims) > target_count:
            # Sort the simulation indices by weight
            sorted_indices = np.argsort([sim.weight for sim in sims])

            # Get the two lowest weight indices to merge
            indices = sorted_indices[:2].tolist()

            # Merge the two lowest weight sims
            sims = self.merge_sims(sims, [indices])

            # Break the loop if the target count is reached
            if len(sims) == target_count:
                break

        return sims

    def split_by_threshold(
        self,
        sims: list[SimMetadata],
        max_allowed_weight: float,
    ) -> list[SimMetadata]:
        """Split the sims by threshold.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations to split.
        max_allowed_weight : float
            The maximum allowed weight for each simulation. If the weight of a
            simulation exceeds this value, it will be split.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after splitting.
        """
        return self.split_by_weight(sims, max_allowed_weight)

    def merge_by_threshold(
        self,
        sims: list[SimMetadata],
        min_allowed_weight: float,
    ) -> list[SimMetadata]:
        """Merge the sims by threshold.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations to merge.
        min_allowed_weight : float
            The minimum allowed weight for each simulation. All the simulations
            with a weight less than this value will be merged into a single
            simulation walker.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after merging.
        """
        while True:
            # Sort the simulations by weight
            sorted_sims = sorted(sims, key=lambda sim: sim.weight)

            # Get the weights of the sorted simulations
            weights = np.array([sim.weight for sim in sorted_sims])

            # Get the simulation indices
            indices = np.arange(len(sims))

            # Find the walkers that need to be merged
            to_merge = indices[weights < min_allowed_weight].tolist()
            if len(to_merge) < 2:  # noqa: PLR2004
                return sims

            # Merge the simulations
            sims = self.merge_sims(sorted_sims, [to_merge])

    def get_pcoords(
        self,
        sims: list[SimMetadata],
        pcoord_idx: int = 0,
    ) -> list[float]:
        """Return the `pcoord_idx` progress coordinate for the simulations."""
        return [sim.parent_pcoord[pcoord_idx] for sim in sims]

    @abstractmethod
    def resample(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Resample the weighted ensemble."""
        ...

    @abstractmethod
    def recycle(self, sims: list[SimMetadata]) -> list[int]:
        """Return a list of simulation indices to recycle."""
        ...


class SplitLowResampler(Resampler):
    """Split the simulation with the lowest progress coordinate."""

    def __init__(  # noqa: PLR0913
        self,
        basis_states: BasisStates,
        num_resamples: int = 1,
        n_split: int = 2,
        target_threshold: float = 0.5,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        n_split : int
            The number of simulations to split each simulation into.
            Default is 2.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state. Default is 0.5.
        pcoord_idx : int
            The index of the progress coordinate to use for splitting and
            merging. Only applicable if a multi-dimensional pcoord is used,
            will choose the specified index of the pcoord for spitting and
            merging. Default is 0.

        """
        super().__init__(basis_states)
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def split(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Split the simulation with the lowest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Find the simulations with the lowest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Split the simulations
        indices = sorted_indices[: self.num_resamples].tolist()

        # Split the simulations
        new_sims = self.split_sims(sims, indices, self.n_split)

        return new_sims

    def merge(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Number of merges is the number of resamples + 1
        # since when we split the simulations we get self.num_resamlpes
        # new simulations and merging them will give us 1 new simulation
        # so we need to merge self.num_resamples + 1 simulations in order
        # to maintain the number of simulations in the ensemble.
        num_merges = self.num_resamples + 1

        # Merge the simulations
        indices = [sorted_indices[-num_merges:].tolist()]

        # Merge the simulations
        new_sims = self.merge_sims(sims, indices)

        return new_sims

    def resample(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Resample the weighted ensemble."""
        # Split the simulations
        simulations = self.split(sims)

        # Merge the simulations
        simulations = self.merge(simulations)

        return simulations

    def recycle(
        self,
        sims: list[SimMetadata],
    ) -> list[int]:
        """Return a list of simulations to recycle."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Recycle the simulations
        return [i for i, p in enumerate(pcoords) if p < self.target_threshold]


class SplitHighResampler(Resampler):
    """Split the simulation with the highest progress coordinate."""

    def __init__(  # noqa: PLR0913
        self,
        basis_states: BasisStates,
        num_resamples: int = 1,
        n_split: int = 2,
        target_threshold: float = 0.5,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        n_split : int
            The number of simulations to split each simulation into.
            Default is 2.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state. Default is 0.5.
        pcoord_idx : int
            The index of the progress coordinate to use for splitting and
            merging. Only applicable if a multi-dimensional pcoord is used,
            will choose the specified index of the pcoord for spitting and
            merging. Default is 0.
        """
        super().__init__(basis_states)
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def split(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Split the simulation with the highest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Split the simulations
        indices = sorted_indices[-self.num_resamples :].tolist()

        # Split the simulations
        new_sims = self.split_sims(sims, indices, self.n_split)

        return new_sims

    def merge(self, sims: list[SimMetadata]) -> list[SimMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(pcoords)

        # Number of merges is the number of resamples + 1
        # since when we split the simulations we get self.num_resamlpes
        # new simulations and merging them will give us 1 new simulation
        # so we need to merge self.num_resamples + 1 simulations in order
        # to maintain the number of simulations in the ensemble.
        num_merges = self.num_resamples + 1

        # Merge the simulations
        indices = [sorted_indices[:num_merges].tolist()]

        # Merge the simulations
        new_sims = self.merge_sims(sims, indices)

        return new_sims

    def resample(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Resample the weighted ensemble."""
        # Split the simulations
        simulations = self.split(sims)

        # Merge the simulations
        simulations = self.merge(simulations)

        return simulations

    def recycle(
        self,
        sims: list[SimMetadata],
    ) -> list[int]:
        """Return a list of simulations to recycle."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Recycle the simulations
        return [i for i, p in enumerate(pcoords) if p > self.target_threshold]


class HuberKimResampler(Resampler):
    """
    Run Huber and Kim resampling.

    The resampling procedure is outlined in Huber, Kim 1996:
        https://doi.org/10.1016/S0006-3495(96)79552-8

    This resampler is designed to be used with bins and follows 3 steps:
        1. Resample based on weight
        2. Adjust the number of simulations in each bin
        3. Split and merge to keep simulations inside weight thresholds
    """

    def __init__(  # noqa: PLR0913
        self,
        basis_states: BasisStates,
        sims_per_bin: int = 5,
        max_allowed_weight: float = 0.25,
        min_allowed_weight: float = 10e-40,
        target_threshold: float = 0.5,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        sims_per_bin : int
            The number of simulations to have in each bin. Default is 5.
        max_allowed_weight : float
            The maximum allowed weight for each simulation. If the weight of a
            simulation exceeds this value, it will be split. Default is 0.25.
        min_allowed_weight : float
            The minimum allowed weight for each simulation. All the simulations
            with a weight less than this value will be merged into a single
            simulation walker. Default is 10e-40.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state. Default is 0.5.
        pcoord_idx : int
            The index of the progress coordinate to use for splitting and
            merging. Only applicable if a multi-dimensional pcoord is used,
            will choose the specified index of the pcoord for spitting and
            merging. Default is 0.
        """
        super().__init__(basis_states)
        self.sims_per_bin = sims_per_bin
        self.max_allowed_weight = max_allowed_weight
        self.min_allowed_weight = min_allowed_weight
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def resample(
        self,
        sims: list[SimMetadata],
    ) -> list[SimMetadata]:
        """Resample the weighted ensemble."""
        # Get the weight of the simulations
        weights = [sim.weight for sim in sims]

        # Calculate the ideal weight
        ideal_weight = sum(weights) / self.sims_per_bin

        # Split the simulations by weight
        sims = self.split_by_weight(sims, ideal_weight)

        # Merge the simulations by weight
        sims = self.merge_by_weight(sims, ideal_weight)

        # Adjust the number of simulations in each bin
        sims = self.adjust_count(sims, self.sims_per_bin)

        # Split the simulations by threshold
        sims = self.split_by_threshold(sims, self.max_allowed_weight)

        # Merge the simulations by threshold
        sims = self.merge_by_threshold(sims, self.min_allowed_weight)

        return sims

    def recycle(
        self,
        sims: list[SimMetadata],
    ) -> list[int]:
        """Return a list of simulations to recycle."""
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Recycle the simulations
        return [i for i, p in enumerate(pcoords) if p < self.target_threshold]
