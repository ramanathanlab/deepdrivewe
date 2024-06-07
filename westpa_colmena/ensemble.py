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


@dataclass
class SimMetadata:
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
    parent_pcoord: list[float] = field(
        metadata={
            'help': 'The progress coordinate for the parent simulation.',
        },
    )
    parent_simulation_id: int | None = field(
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
    pcoord: list[float] | None = field(
        default=None,
        metadata={
            'help': 'The progress coordinate for the simulation.',
        },
    )

    def __hash__(self) -> int:
        """Hash the simulation metadata to ensure that it is unique."""
        return hash((self.simulation_id, self.restart_file))

    def copy(self) -> SimMetadata:
        """Return a copy of the simulation metadata."""
        return SimMetadata(**self.__dict__)


# TODO: Figure out how to initialize basis state parent_pcoord
class BasisStates(ABC):
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

        # Compute the pcoord for each basis state
        basis_pcoords = [
            self.init_basis_pcoord(basis_file) for basis_file in basis_files
        ]

        # Initialize the basis states
        self.basis_states = self._uniform_init(basis_files, basis_pcoords)

        # Log the number of basis states
        print(f'Loaded {len(self.basis_states)} basis states')

    def __len__(self) -> int:
        """Return the number of basis states."""
        return len(self.basis_states)

    def __getitem__(self, idx: int) -> SimMetadata:
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
        basis_pcoords: list[list[float]],
    ) -> list[SimMetadata]:
        # Assign a uniform weight to each of the basis states
        weight = 1.0 / self.ensemble_members

        # Create the metadata for each basis state to populate the
        # first iteration
        simulations = []
        for idx, (f, p) in enumerate(zip(basis_files, basis_pcoords)):
            simulations.append(
                SimMetadata(
                    weight=weight,
                    simulation_id=idx,
                    iteration_id=0,
                    parent_restart_file=f,
                    parent_pcoord=p,
                ),
            )

        return simulations

    @abstractmethod
    def init_basis_pcoord(self, basis_file: Path) -> list[float]:
        """Initialize the progress coordinate for a basis state."""
        ...


class WeightedEnsemble:
    """Weighted ensemble."""

    # The list of simulations for each iteration
    simulations: list[list[SimMetadata]]

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
        checkpoint_name = f'weighted_ensemble-itr-{len(self.simulations):06d}'
        checkpoint_file = output_dir / f'{checkpoint_name}.pkl'

        # Save the weighted ensemble to the checkpoint file
        with checkpoint_file.open('wb') as f:
            pickle.dump(self, f)

    @property
    def current_iteration(self) -> list[SimMetadata]:
        """Return the simulations for the current iteration."""
        return self.simulations[-1]

    def advance_iteration(
        self,
        next_iteration: list[SimMetadata],
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
