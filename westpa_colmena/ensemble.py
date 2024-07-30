"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
import pickle
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel
from pydantic import Field


class TargetState(BaseModel):
    """Target state for the weighted ensemble."""

    label: str = Field(
        '',
        description='The label for the target state.',
    )
    pcoord: list[float] = Field(
        ...,
        description='The progress coordinate for the target state.',
    )


class IterationMetadata(BaseModel):
    """Metadata for an iteration in the weighted ensemble."""

    iteration_id: int = Field(
        ...,
        description='The ID of the iteration.',
    )
    binner_pickle: bytes = Field(
        default='',
        description='The pickled binner used to assign simulations.',
    )
    binner_hash: str = Field(
        default='',
        description='The hash of the binner used to assign simulations.',
    )
    min_bin_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description='The minimum bin probability for an iteration.',
    )
    max_bin_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description='The maximum bin probability for an iteration.',
    )
    bin_target_counts: list[int] = Field(
        default_factory=list,
        description='The target counts for each bin.',
    )


class SimMetadata(BaseModel):
    """Metadata for a simulation in the weighted ensemble."""

    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='The weight of the simulation.',
    )
    simulation_id: int = Field(
        ...,
        description='The ID of the simulation.',
    )
    iteration_id: int = Field(
        ...,
        description='The ID of the iteration the simulation is in.',
    )
    parent_restart_file: Path = Field(
        ...,
        description='The restart file for the parent simulation.',
    )
    parent_pcoord: list[float] = Field(
        ...,
        description='The progress coordinate for the parent simulation.',
    )
    parent_simulation_id: int | None = Field(
        default=None,
        description='The ID of the previous simulation the current one is'
        " split from, or None if it's a basis state.",
    )
    wtg_parent_ids: list[int] = Field(
        default_factory=list,
        description='The IDs of the parent simulation(s) to compute the '
        'weight graph. This accounts for merged simulations.',
    )
    restart_file: Path | None = Field(
        default=None,
        description='The restart file for the simulation.',
    )
    pcoord: list[list[float]] = Field(
        default_factory=list,
        description='The progress coordinates for the simulation. '
        'Shape: (n_frames, pcoord_dim). where n_frames is the number of '
        'frames in the trajectory and pcoord_dim is the dimension of the '
        'progress coordinate.',
    )
    auxdata: dict[str, list[int | float]] = Field(
        default_factory=dict,
        description='Auxiliary data for the simulation (stores auxiliary '
        'pcoords, etc). Does not store raw coords since that would create '
        'a bottleneck when writing the HDF5 file.',
    )
    endpoint_type: int = Field(
        default=1,
        description='The type of endpoint for the simulation. Default is 1.'
        '1 indicates the simulation should continue, 2 indicates the '
        'simulation ended in a merge, and 3 indicates the simulation '
        'ended by recycling.',
    )
    cputime: float = Field(
        default=0.0,
        ge=0.0,
        description='The CPU time for the simulation (i.e., clock time).',
    )
    walltime: float = Field(
        default=0.0,
        ge=0.0,
        description='The wall time for the simulation (i.e., system wide).',
    )

    # TODO: Do we still need this?
    def __hash__(self) -> int:
        """Hash the simulation metadata to ensure that it is unique."""
        return hash((self.simulation_id, self.restart_file))


class BasisStates(ABC):
    """Basis states for the weighted ensemble."""

    def __init__(
        self,
        simulation_input_dir: Path,
        basis_state_ext: str,
        initial_ensemble_members: int,
    ) -> None:
        """Initialize the basis states.

        Parameters
        ----------
        simulation_input_dir : Path
            The directory containing the simulation input files.
        basis_state_ext : str
            The extension of the basis state files.
        initial_ensemble_members : int
            The number of initial ensemble members.
        """
        self.simulation_input_dir = simulation_input_dir
        self.basis_state_ext = basis_state_ext
        self.ensemble_members = initial_ensemble_members

        # Load the basis states
        basis_files = self._load_basis_states()

        # Compute the pcoord for each basis state
        basis_pcoords = [
            self.init_basis_pcoord(basis_file) for basis_file in basis_files
        ]

        # Initialize the basis states
        self.basis_states = self._uniform_init(basis_files, basis_pcoords)

        # Store the unique basis states (to be used in the HDF5 I/O module)
        self.unique_basis_states = self.basis_states[: len(basis_files)]

        # Log the number of basis states
        print(f'Loaded {len(self.basis_states)} basis states')

    def __len__(self) -> int:
        """Return the number of basis states."""
        return len(self.basis_states)

    def __getitem__(self, idx: int) -> SimMetadata:
        """Return the basis state at the specified index."""
        return self.basis_states[idx]

    def __iter__(self) -> Iterator[SimMetadata]:
        """Return an iterator over the basis states."""
        return iter(self.basis_states)

    def _load_basis_states(self) -> list[Path]:
        """Load the unique basis states from the simulation input directory.

        Returns
        -------
        list[Path]
            The unique basis state paths in the simulation input directory.

        Raises
        ------
        FileNotFoundError
            If no basis state is found in the input directory.
        """
        # Collect initial simulation directories,
        # assuming they are in nested subdirectories
        sim_input_dirs = [
            p for p in self.simulation_input_dir.glob('*') if p.is_dir()
        ]

        # Check if there are more input dirs than initial ensemble members
        sim_input_dirs = sim_input_dirs[: self.ensemble_members]

        # Get the basis states by globbing the input directories
        basis_states = []
        for input_dir in sim_input_dirs:
            # Define the glob pattern
            pattern = f'*{self.basis_state_ext}'

            # Get the basis state file in the input directory
            basis_state = next(input_dir.glob(pattern), None)

            # Raise an error if no basis state is found
            if basis_state is None:
                raise FileNotFoundError(
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

        # Create a index map to get a unique id for each basis state
        # (note we add 1 to the index to avoid a parent ID of 0)
        index_map = {file: -(idx + 1) for idx, file in enumerate(basis_files)}

        # Create the metadata for each basis state to populate the
        # first iteration. We cycle/repeat through the basis state
        # files to the desired number of ensemble members.
        simulations = []
        for idx, (file, pcoord) in zip(
            range(self.ensemble_members),
            itertools.cycle(zip(basis_files, basis_pcoords)),
        ):
            simulations.append(
                SimMetadata(
                    weight=weight,
                    simulation_id=idx,
                    iteration_id=0,
                    parent_restart_file=file,
                    parent_pcoord=pcoord,
                    # Set the parent simulation ID to the negative of the
                    # index to indicate that the simulation is a basis state
                    # (note we add 1 to the index to avoid a parent ID of 0)
                    parent_simulation_id=index_map[file],
                    # TODO: Can we double check this is correct?
                    wtg_parent_ids=[index_map[file]],
                ),
            )

        return simulations

    @abstractmethod
    def init_basis_pcoord(self, basis_file: Path) -> list[float]:
        """Initialize the progress coordinate for a basis state."""
        ...


# TODO: Unify this with the westh5 file since that is the checkpoint
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
