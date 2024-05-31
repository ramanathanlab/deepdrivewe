"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
import pickle
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
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

        # Log the number of basis states
        print(f'Loaded {len(self.basis_states)} basis states')

    def __len__(self) -> int:
        """Return the number of basis states."""
        return len(self.basis_states)

    def __getitem__(self, idx: int) -> SimulationMetadata:
        """Return the basis state at the specified index."""
        return self.basis_states[idx]

    def _load_basis_states(self) -> list[Path]:
        # Collect initial simulation directories, assumes they are in nested
        # subdirectories
        simulation_input_dirs = filter(
            lambda p: p.is_dir(),
            self.simulation_input_dir.glob('*'),
        )

        # Get the basis states by globbing the input directories
        basis_states = []
        for _, input_dir in zip(
            range(self.ensemble_members),
            itertools.cycle(simulation_input_dirs),
        ):
            # Define the glob pattern
            pattern = f'*{self.basis_state_ext}'

            # Get the basis state file in the input directory
            basis_state = next(input_dir.glob(pattern), None)

            # Raise an error if no basis state is found
            if basis_state is None:
                raise ValueError(
                    f'No basis state in {input_dir} found with'
                    f' extension: {self.basis_state_ext}',
                )

            # Append the basis state to the list of basis states
            basis_states.append(basis_state)

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

    # The list of simulations for each iteration
    simulations: list[list[SimulationMetadata]]

    def __init__(
        self,
        basis_states: BasisStates,
        checkpoint_dir: Path | None = None,
        resume_checkpoint: Path | None = None,
    ) -> None:
        """Initialize the weighted ensemble.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        checkpoint_dir : Path or None
            The directory to save the weighted ensemble checkpoints.
        resume_checkpoint : Path or None
            The checkpoint file to resume the weighted ensemble from.
        """
        self.basis_states = basis_states
        self.checkpoint_dir = checkpoint_dir

        # Initialize the checkpoint dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the weighted ensemble simulations
        if resume_checkpoint is None:
            # Initialize the ensemble with the basis states
            self.simulations = [deepcopy(self.basis_states.basis_states)]
        else:
            # Load the weighted ensemble from the checkpoint file
            self.load_checkpoint(resume_checkpoint)

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

        # Save the weighted ensemble to a checkpoint file
        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir)


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
        print(f'{indices=}')
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
