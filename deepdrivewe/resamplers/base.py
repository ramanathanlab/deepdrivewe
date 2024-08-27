"""Resampling algorithms for the weighted ensemble."""

from __future__ import annotations

import itertools
import math
from abc import ABC
from abc import abstractmethod

import numpy as np

from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.binners.base import Binner
from deepdrivewe.recyclers.base import Recycler


class Resampler(ABC):
    """Resampler for the weighted ensemble."""

    def __init__(self) -> None:
        """Initialize the resampler."""
        self._index_counter = itertools.count()

    def run(
        self,
        cur_sims: list[SimMetadata],
        binner: Binner,
        recycler: Recycler,
    ) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
        """Assign simulations to bins and resample the weighted ensemble.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        binner : Binner
            The binner to use for binning the simulations.
        recycler : Recycler
            The recycler to use for recycling the simulations.

        Returns
        -------
        tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]
            The current sims, the new sims, and the iteration metadata.
        """
        # Get the next iteration of simulation metadata
        next_sims = self._get_next_sims(cur_sims)

        # Recycle the current iteration
        cur_sims, next_sims = recycler.recycle_simulations(cur_sims, next_sims)

        # Assign the simulations to bins
        bin_assignments = binner.bin_simulations(next_sims)

        # Compute the iteration metadata
        metadata = binner.compute_iteration_metadata(cur_sims)

        # Resample the simulations in each bin
        new_sims = []
        for bin_sims in bin_assignments.values():
            # Get the simulations in the bin
            binned_sims = [next_sims[sim_idx] for sim_idx in bin_sims]

            # Resample the bin and add them to the new simulations
            cur_sims, resampled_sims = self.resample(cur_sims, binned_sims)

            # Add the resampled simulations to the new simulations
            new_sims.extend(resampled_sims)

        return cur_sims, new_sims, metadata

    def _get_next_sims(self, cur_sims: list[SimMetadata]) -> list[SimMetadata]:
        """Return the simulations for the next iteration."""
        # Create a list to store the new simulations for this iteration
        simulations = []

        for idx, sim in enumerate(cur_sims):
            # Ensure that the simulation has a restart file, i.e., the `sim`
            # object represents a simulation that has been run.
            assert sim.restart_file is not None

            # Create the metadata for the new simulation
            new_sim = SimMetadata(
                weight=sim.weight,
                simulation_id=idx,
                iteration_id=sim.iteration_id + 1,
                parent_restart_file=sim.restart_file,
                # The parent progress coordinate is the progress coordinate
                # of the last frame of the previous simulation
                parent_pcoord=sim.pcoord[-1],
                parent_simulation_id=sim.simulation_id,
                wtg_parent_ids=[sim.simulation_id],
            )

            # Add the new simulation to the current iteration
            simulations.append(new_sim)

        return simulations

    def _add_new_simulation(
        self,
        sim: SimMetadata,
        weight: float,
        wtg_parent_ids: list[int],
    ) -> SimMetadata:
        """Add a new simulation to the current iteration."""
        # Create the metadata for the new simulation
        return SimMetadata(
            weight=weight,
            simulation_id=next(self._index_counter),
            iteration_id=sim.iteration_id,
            restart_file=sim.restart_file,
            parent_restart_file=sim.parent_restart_file,
            parent_pcoord=sim.parent_pcoord,
            parent_simulation_id=sim.parent_simulation_id,
            wtg_parent_ids=wtg_parent_ids,
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
                # NOTE: The split simulation is assigned a weight equal to
                # the original weight divided by the number of splits. It
                # also inherits the previous wtg_parent_ids.
                new_sim = self._add_new_simulation(
                    sim,
                    sim.weight / n_split,
                    sim.wtg_parent_ids,
                )
                new_sims.append(new_sim)

        return new_sims

    def merge_sims(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        indices: list[int],
    ) -> list[SimMetadata]:
        """Merge each group of simulation indices into a single simulation.

        NOTE: This method modifies cur sims in place to set the endpoint type.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
            The list of next simulations in a particular bin to merge.
        indices : list[int]
            The indices of the next simulations to merge.

        Returns
        -------
        list[SimMetadata]
            The list of new simulations after merging.
        """
        # Get the simulations to merge
        to_merge = [next_sims[idx] for idx in indices]

        # Get the weights of each simulation to merge
        weights = [sim.weight for sim in to_merge]

        # Make sure the weights are normalized to sum to 1 for randomizing.
        # Since the entire ensemble should have a total weight of 1
        # any subset of the ensemble will have a total weight less than 1.
        norm_weights = np.array(weights) / sum(weights)

        # Randomly select one of the simulations with probability equal
        # to the normalized weights
        select: int = np.random.choice(len(to_merge), p=norm_weights)

        # Compute the union of all the wtg_parent_ids
        all_wtg_parent_ids = [set(sim.wtg_parent_ids) for sim in to_merge]
        wtg_parent_ids = list(set.union(*all_wtg_parent_ids))

        # Add the new simulation to the current iteration
        new_sim = self._add_new_simulation(
            to_merge[select],
            sum(weights),
            wtg_parent_ids,
        )

        # Create a list to store the new simulations
        new_sims: list[SimMetadata] = []

        # Get the indices of non-merged simulations
        no_merge_idxs = [i for i in range(len(next_sims)) if i not in indices]

        # Add back the simulations that will not be merged
        new_sims.extend(next_sims[i] for i in no_merge_idxs)

        # Add the new simulation to the list of new simulations
        new_sims.append(new_sim)

        # Get the parent simulation IDs of all the merged simulations
        merged_parents = {x.parent_simulation_id for x in to_merge}
        # Remove the parent simulation id of the new merged simulation
        merged_parents.remove(new_sim.parent_simulation_id)

        # Set the endpoint type for the merged simulations (except the new sim)
        for sim in cur_sims:
            # sim.simulation_id >= 0 ensures that the simulation has not
            # been recycled (i.e., it is not a negative index)
            if sim.simulation_id >= 0 and sim.simulation_id in merged_parents:
                # Set the endpoint type to 2 if the simulation is merged
                sim.endpoint_type = 2

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
            The list of new simulations after splitting.
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
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        ideal_weight: float,
    ) -> list[SimMetadata]:
        """Merge underweight sims.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
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
            sorted_sims = sorted(next_sims, key=lambda sim: sim.weight)

            # Get the weights of the sorted simulations
            weights = np.array([sim.weight for sim in sorted_sims])

            # Accumulate the weights
            cumul_weight = np.add.accumulate(weights)

            # Get the simulation indices
            indices = np.arange(len(next_sims))

            # Find the walkers that need to be merged
            to_merge = indices[cumul_weight <= ideal_weight].tolist()

            # Break the loop if no walkers need to be merged
            if len(to_merge) < 2:  # noqa: PLR2004
                return next_sims

            # Merge the simulations
            next_sims = self.merge_sims(cur_sims, sorted_sims, to_merge)

    def adjust_count(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        target_count: int,
    ) -> list[SimMetadata]:
        """Adjust the number of sims to match the target count.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
            The list of simulations in a particular bin to adjust.
        target_count : int
            The number of simulations to have in the bin.

        Returns
        -------
        list[SimMetadata]
            The list of simulations after adjusting.
        """
        # Case 1: Too few sims
        while len(next_sims) < target_count:
            # Get the index of the largest weight simulation
            index = int(np.argmax([sim.weight for sim in next_sims]))

            # Split the highest weight sim in two
            next_sims = self.split_sims(next_sims, [index], 2)

            # Break the loop if the target count is reached
            if len(next_sims) == target_count:
                break

        # Case 2: Too many sims
        while len(next_sims) > target_count:
            # Sort the simulation indices by weight
            sorted_indices = np.argsort([sim.weight for sim in next_sims])

            # Get the two lowest weight indices to merge
            indices = sorted_indices[:2].tolist()

            # Merge the two lowest weight sims
            next_sims = self.merge_sims(cur_sims, next_sims, indices)

            # Break the loop if the target count is reached
            if len(next_sims) == target_count:
                break

        return next_sims

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
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        min_allowed_weight: float,
    ) -> list[SimMetadata]:
        """Merge all simulations under a given threshold into a single sim.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
            The list of simulations in a particular bin to merge.
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
            sorted_sims = sorted(next_sims, key=lambda sim: sim.weight)

            # Get the weights of the sorted simulations
            weights = np.array([sim.weight for sim in sorted_sims])

            # Get the simulation indices
            indices = np.arange(len(next_sims))

            # Find the walkers that need to be merged
            to_merge = indices[weights < min_allowed_weight].tolist()
            if len(to_merge) < 2:  # noqa: PLR2004
                return next_sims

            # Merge the simulations
            next_sims = self.merge_sims(cur_sims, sorted_sims, to_merge)

    def get_pcoords(
        self,
        next_sims: list[SimMetadata],
        pcoord_idx: int = 0,
    ) -> list[float]:
        """Extract the progress coordinates from the simulations.

        Parameters
        ----------
        next_sims : list[SimMetadata]
            The list of simulation metadata.
        pcoord_idx : int
            The index of the progress coordinate to extract. Default is 0.

        Returns
        -------
        list[float]
            The progress coordinates for the simulations.
        """
        return [sim.parent_pcoord[pcoord_idx] for sim in next_sims]

    @abstractmethod
    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Resample the weighted ensemble."""
        ...
