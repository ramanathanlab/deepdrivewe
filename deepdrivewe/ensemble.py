"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Iterator
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field


class IterationMetadata(BaseModel):
    """Metadata for an iteration in the weighted ensemble."""

    iteration_id: int = Field(
        default=1,
        ge=1,
        description='The ID of the iteration (1-indexed).',
    )
    binner_pickle: bytes = Field(
        default='',
        description='The pickled binner used to assign simulations.',
        exclude=True,  # Exclude from JSON serialization to checkpoint
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
        ge=1,
        description='The ID of the iteration the simulation is in '
        '(1-indexed).',
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
    auxdata: dict[str, list[list[int | float]]] = Field(
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


class BasisStateInitializer(Protocol):
    """Protocol for initializing the progress coordinate for a basis state."""

    def __call__(self, basis_file: str) -> list[float]:
        """Initialize the progress coordinate for a basis state."""
        ...


class BasisStates(BaseModel):
    """Basis states for the weighted ensemble."""

    basis_state_dir: Path = Field(
        description='Nested directory storing initial simulation start files, '
        'e.g. pdb_dir/system1/, pdb_dir/system2/, ..., where system<i> might '
        'store PDB files, topology files, etc needed to start the simulation '
        'application.',
    )
    basis_state_ext: str = Field(
        default='.ncrst',
        description='Extension for the basis states.',
    )
    initial_ensemble_members: int = Field(
        ...,
        ge=1,
        description='The number of initial ensemble members.',
    )
    num_basis_files: int = Field(
        default=0,
        description='The number of unique basis state files.',
    )
    basis_states: list[SimMetadata] = Field(
        default_factory=list,
        description='The basis states for the weighted ensemble.',
    )

    @property
    def unique_basis_states(self) -> list[SimMetadata]:
        """Return the unique basis states."""
        # (to be used in the HDF5 I/O module)
        return self.basis_states[: self.num_basis_files]

    def __len__(self) -> int:
        """Return the number of basis states."""
        return len(self.basis_states)

    def __getitem__(self, idx: int) -> SimMetadata:
        """Return the basis state at the specified index."""
        return self.basis_states[idx]

    def __iter__(self) -> Iterator[SimMetadata]:
        """Return an iterator over the basis states."""
        return iter(self.basis_states)

    def load_basis_states(
        self,
        basis_state_initializer: BasisStateInitializer,
    ) -> None:
        """Load the basis states for the weighted ensemble."""
        # Collect the basis state files
        basis_files = self._glob_basis_states()

        # Compute the pcoord for each basis state
        basis_pcoords = [
            basis_state_initializer(basis_file.as_posix())
            for basis_file in basis_files
        ]

        # Initialize the basis states
        self.basis_states = self._uniform_init(basis_files, basis_pcoords)

        # Set the number of basis files
        self.num_basis_files = len(basis_files)

        # Log the number of basis states
        print(f'Loaded {len(self.basis_states)} basis states')

    def _glob_basis_states(self) -> list[Path]:
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
            p for p in self.basis_state_dir.glob('*') if p.is_dir()
        ]

        # Check if there are more input dirs than initial ensemble members
        sim_input_dirs = sim_input_dirs[: self.initial_ensemble_members]

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
        weight = 1.0 / self.initial_ensemble_members

        # Create a index map to get a unique id for each basis state
        # (note we add 1 to the index to avoid a parent ID of 0)
        index_map = {file: -(idx + 1) for idx, file in enumerate(basis_files)}

        # Create the metadata for each basis state to populate the
        # first iteration. We cycle/repeat through the basis state
        # files to the desired number of ensemble members.
        simulations = []
        for idx, (file, pcoord) in zip(
            range(self.initial_ensemble_members),
            itertools.cycle(zip(basis_files, basis_pcoords)),
        ):
            simulations.append(
                SimMetadata(
                    weight=weight,
                    simulation_id=idx,
                    iteration_id=1,  # WESTPA is 1-indexed
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


class WeightedEnsemble(BaseModel):
    """Weighted ensemble."""

    basis_states: BasisStates = Field(
        ...,
        description='The basis states for the weighted ensemble.',
    )
    target_states: list[TargetState] = Field(
        ...,
        description='The target states for the weighted ensemble.',
    )
    simulations: list[list[SimMetadata]] = Field(
        default_factory=list,
        description='The list of simulations for each iteration.',
    )
    metadata: IterationMetadata = Field(
        default=IterationMetadata,
        description='The metadata for the current iteration.',
    )
    cur_sims: list[SimMetadata] = Field(
        default_factory=list,
        description='The simulations for the current iteration.',
    )

    def initialize_basis_states(
        self,
        basis_state_initializer: BasisStateInitializer,
    ) -> None:
        """Load the basis states for the weighted ensemble.

        Parameters
        ----------
        basis_state_initializer : BasisStateInitializer
            The initializer for the basis states (e.g., a function that
            reads the progress coordinate from a file and computes and
            returns a progress coordinate).
        """
        # Load the basis states
        self.basis_states.load_basis_states(basis_state_initializer)

        # Initialize the simulations with the basis states
        self.simulations = [deepcopy(self.basis_states.basis_states)]

    @property
    def current_sims(self) -> list[SimMetadata]:
        """Return the simulations for the current iteration."""
        return self.simulations[-1]

    @property
    def iteration(self) -> int:
        """Return the current iteration of the weighted ensemble."""
        # The first iteration is the basis states
        return len(self.simulations) - 1

    def advance_iteration(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        metadata: IterationMetadata,
    ) -> None:
        """Advance the iteration of the weighted ensemble.

        The binner is responsible for determining which simulations to split
        and merge. The binner will then call this method to advance the
        iteration of the weighted ensemble.
        """
        # Create a list to store the new simulations for this iteration
        self.simulations.append(next_sims)
        self.metadata = metadata
        self.cur_sims = cur_sims
