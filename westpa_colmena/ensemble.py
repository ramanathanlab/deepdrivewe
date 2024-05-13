"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
import pickle
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np


@dataclass
class SimulationMetadata:
    """Metadata for a simulation in the weighted ensemble."""

    weight: float = field(
        metadata={
            'help': 'The weight of the simulation.',
        },
    )
    simulation_id: int = field(
        metadata={
            'help': 'The ID of the simulation.',
        },
    )
    iteration_id: int = field(
        metadata={
            'help': 'The ID of the iteration the simulation is in.',
        },
    )
    parent_restart_file: Path = field(
        metadata={
            'help': 'The restart file for the parent simulation.',
        },
    )
    prev_simulation_id: int | None = field(
        default=None,
        metadata={
            'help': 'The ID of the previous simulation the current one is'
            " split from, or None if it's a basis state.",
        },
    )
    restart_file: Path | None = field(
        default=None,
        metadata={
            'help': 'The restart file for the simulation.',
        },
    )

    def __hash__(self) -> int:
        """Hash the simulation metadata to ensure that it is unique."""
        return hash((self.simulation_id, self.restart_file))

    def copy(self) -> SimulationMetadata:
        """Return a copy of the simulation metadata."""
        return SimulationMetadata(**self.__dict__)


class BasisStates:
    """Basis states for the weighted ensemble."""

    def __init__(
        self,
        simulation_input_dir: Path,
        basis_state_ext: str,
        ensemble_members: int,
    ) -> None:
        """Initialize the basis states.

        Parameters
        ----------
        simulation_input_dir : Path
            The directory containing the simulation input files.
        basis_state_ext : str
            The extension of the basis state files.
        """
        self.simulation_input_dir = simulation_input_dir
        self.basis_state_ext = basis_state_ext
        self.ensemble_members = ensemble_members

        # Load the basis states
        basis_files = self._load_basis_states()

        # Initialize the basis states
        self.basis_states = self._uniform_init(basis_files)

    def __len__(self) -> int:
        """Return the number of basis states."""
        return len(self.basis_states)

    def __getitem__(self, idx: int) -> SimulationMetadata:
        """Return the basis state at the specified index."""
        return self.basis_states[idx]

    def _load_basis_states(self) -> list[Path]:
        # Collect initial simulation directories, assumes they are in nested
        # subdirectories
        simulation_input_dirs = itertools.cycle(
            filter(lambda p: p.is_dir(), self.simulation_input_dir.glob('*')),
        )

        # Get the basis states by globbing the input directories
        basis_states = [
            next(p.glob(self.basis_state_ext)) for p in simulation_input_dirs
        ]

        # Get the first `ensemble_members` basis states
        basis_states = basis_states[: self.ensemble_members]

        # Return the basis states
        return basis_states

    def _uniform_init(
        self,
        basis_files: list[Path],
    ) -> list[SimulationMetadata]:
        # Assign a uniform weight to each of the basis states
        weight = 1.0 / self.ensemble_members

        # Create the metadata for each basis state to populate the
        # first iteration
        simulations = [
            SimulationMetadata(
                weight=weight,
                simulation_id=idx,
                prev_simulation_id=None,
                iteration_id=0,
                restart_file=None,
                parent_restart_file=basis_file,
            )
            for idx, basis_file in enumerate(basis_files)
        ]

        return simulations


class WeightedEnsemble:
    """Weighted ensemble."""

    # TODO: Figure out a checkpointing mechanism for the metadata

    # The list of simulations for each iteration
    simulations: list[list[SimulationMetadata]]

    def __init__(
        self,
        basis_states: BasisStates,
        checkpoint_file: Path | None,
    ) -> None:
        """Initialize the weighted ensemble.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        """
        self.basis_states = basis_states

        # If checkpoint file is provided, load the weighted ensemble
        if checkpoint_file is not None:
            self.load_checkpoint(checkpoint_file)

    def load_checkpoint(self, checkpoint_file: Path) -> None:
        """Load the weighted ensemble from a checkpoint file."""
        # Load the weighted ensemble from the checkpoint file
        with checkpoint_file.open('rb') as f:
            weighted_ensemble = pickle.load(f)

        # Update the weighted ensemble
        self.__dict__.update(weighted_ensemble.__dict__)

    def save_checkpoint(self, output_dir: Path) -> None:
        """Save the weighted ensemble to a checkpoint file."""
        # Make the output directory if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the weighted ensemble to a checkpoint file
        checkpoint_name = f'weighted_ensemble-itr-{len(self.simulations)::06d}'
        checkpoint_file = output_dir / f'{checkpoint_name}.pkl'

        # Save the weighted ensemble to the checkpoint file
        with checkpoint_file.open('wb') as f:
            pickle.dump(self, f)

    @property
    def current_iteration(self) -> list[SimulationMetadata]:
        """Return the simulations for the current iteration."""
        return self.simulations[-1]

    def advance_iteration(
        self,
        next_iteration: list[SimulationMetadata],
    ) -> None:
        """Advance the iteration of the weighted ensemble.

        The binner is responsible for determining which simulations to split
        and merge. The binner will then call this method to advance the
        iteration of the weighted ensemble.
        """
        # Create a list to store the new simulations for this iteration
        self.simulations.append(next_iteration)


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
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
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
                prev_simulation_id = -1 * sim.simulation_id
            else:
                # If the simulation is not recycled, set the parent restart
                # file and simulation id to the restart file of the current
                # simulation
                parent_restart_file = sim.restart_file
                prev_simulation_id = sim.simulation_id

            # Create the metadata for the new simulation
            new_sim = SimulationMetadata(
                weight=sim.weight,
                simulation_id=idx,
                iteration_id=sim.iteration_id + 1,
                prev_simulation_id=prev_simulation_id,
                restart_file=None,
                parent_restart_file=parent_restart_file,
            )

            # Add the new simulation to the current iteration
            simulations.append(new_sim)

        return simulations

    def _add_new_simulation(
        self,
        sim: SimulationMetadata,
        weight: float,
    ) -> SimulationMetadata:
        """Add a new simulation to the current iteration."""
        # Create the metadata for the new simulation
        return SimulationMetadata(
            weight=weight,
            simulation_id=next(self.index_counter),
            iteration_id=sim.iteration_id,
            prev_simulation_id=sim.prev_simulation_id,
            restart_file=sim.restart_file,
            parent_restart_file=sim.parent_restart_file,
        )

    def split_sims(
        self,
        sims: list[SimulationMetadata],
        indices: list[int],
        n_splits: int | list[int] = 2,
    ) -> list[SimulationMetadata]:
        """Split the simulation index into `n_split`."""
        # Get the simulations to split
        sims_to_split = [sims[idx] for idx in indices]

        # Handle the case where `n_split` is a single integer
        if isinstance(n_splits, int):
            n_splits = [n_splits] * len(sims_to_split)

        # Create a list to store the new simulations
        new_sims: list[SimulationMetadata] = []

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
        sims: list[SimulationMetadata],
        indices: list[list[int]],
    ) -> list[SimulationMetadata]:
        """Merge each group of simulation indices into a single simulation."""
        # Get the indices of non-merged simulations
        merge_idxs = [idx for index_group in indices for idx in index_group]
        no_merge_idxs = [i for i in range(len(sims)) if i not in merge_idxs]

        # Create a list to store the new simulations
        new_sims: list[SimulationMetadata] = []

        # Add back the simulations that will not be merged
        new_sims.extend(sims[i] for i in no_merge_idxs)

        for index_group in indices:
            # Get the simulations to merge
            to_merge = [sims[idx] for idx in index_group]

            # Get the weights of each simulation to merge
            weights = [sim.weight for sim in to_merge]

            # Randomly select one of the simulations weighted by their weights
            select: int = np.random.choice(len(to_merge), p=weights)

            # Add the new simulation to the current iteration
            new_sim = self._add_new_simulation(to_merge[select], sum(weights))

            # Add the new simulation to the list of new simulations
            new_sims.append(new_sim)

        # Return the new simulation
        return new_sims

    @abstractmethod
    def resample(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Resample the weighted ensemble."""
        ...

    @abstractmethod
    def recycle(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[int]:
        """Return a list of simulation indices to recycle."""
        ...


class NaiveResampler(Resampler):
    """Naive resampler."""

    def __init__(  # noqa: PLR0913
        self,
        pcoord: list[float],
        num_resamples: int = 1,
        n_split: int = 2,
        split_low: bool = True,
        target_threshold: float = 0.5,
    ) -> None:
        """Initialize the resampler.

        Parameters
        ----------
        pcoord : list[float]
            The progress coordinate for the simulations.
        num_resamples : int
            The number of resamples to perform (i.e., the number of splits
            and merges to perform in each iteration). Default is 1.
        n_split : int
            The number of simulations to split each simulation into.
            Default is 2.
        split_low : bool
            If True, split the simulation with the lowest progress coordinate
            and merge the simulations with the highest progress coordinate.
            If False, split the simulation with the highest progress coordinate
            and merge the simulations with the lowest progress coordinate.
            Default is True.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state. Default is 0.5.
        """
        self.pcoord = pcoord
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.split_low = split_low
        self.target_threshold = target_threshold

    def split(
        self,
        simulations: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Split the simulation with the lowest progress coordinate."""
        # Find the simulations with the lowest progress coordinate
        sorted_indices = np.argsort(self.pcoord)

        # Split the simulations
        if self.split_low:
            indices = list(sorted_indices[: self.num_resamples])
        else:
            indices = list(sorted_indices[-self.num_resamples :])

        # Split the simulations
        new_sims = self.split_sims(simulations, indices, self.n_split)

        return new_sims

    def merge(
        self,
        simulations: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Merge the simulations with the highest progress coordinate."""
        # Find the simulations with the highest progress coordinate
        sorted_indices = np.argsort(self.pcoord)

        # Merge the simulations
        if self.split_low:
            indices = list(sorted_indices[-self.num_resamples :])
        else:
            indices = list(sorted_indices[: self.num_resamples])

        # Merge the simulations
        new_sims = self.merge_sims(simulations, [indices])

        return new_sims

    def resample(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Resample the weighted ensemble."""
        # Generate the next iteration
        simulations = self.get_next_iteration(current_iteration)

        # Split the simulations
        simulations = self.split(simulations)

        # Merge the simulations
        simulations = self.merge(simulations)

        return simulations

    def recycle(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[int]:
        """Return a list of simulations to recycle."""
        # Recycle the simulations
        if self.split_low:
            indices = [
                i
                for i, p in enumerate(self.pcoord)
                if p < self.target_threshold
            ]
        else:
            indices = [
                i
                for i, p in enumerate(self.pcoord)
                if p > self.target_threshold
            ]

        return indices
