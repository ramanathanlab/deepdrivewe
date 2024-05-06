"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
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


class WeightedEnsemble:
    """Weighted ensemble."""

    # TODO: Figure out a checkpointing mechanism for the metadata

    # The list of simulations for each iteration
    simulations: list[list[SimulationMetadata]]

    def __init__(
        self,
        ensemble_members: int,
        simulation_input_dir: Path,
        basis_state_ext: str,
    ) -> None:
        """Initialize the weighted ensemble.

        Parameters
        ----------
        ensemble_members : int
            The number of simulations to start the weighted ensemble with.
        simulation_input_dir : Path
            The directory containing the simulation input files.
        basis_state_ext : str
            The extension of the basis state files.
        """
        self.ensemble_members = ensemble_members
        self.simulation_input_dir = simulation_input_dir
        self.basis_state_ext = basis_state_ext

        # The list of simulations for each iteration
        self.simulations = []
        # Assign a uniform weight to each of the basis states
        weight = 1.0 / ensemble_members

        # Load the basis states
        basis_states = self._load_basis_states()

        # Create the metadata for each basis state to populate the
        # first iteration
        sims = [
            SimulationMetadata(
                weight=weight,
                simulation_id=idx,
                prev_simulation_id=None,
                restart_file=None,
                parent_restart_file=basis_state,
            )
            for idx, basis_state in enumerate(basis_states)
        ]

        self.simulations.append(sims)

    def _load_basis_states(self) -> list[Path]:
        # Collect initial simulation directories, assumes they are in nested
        # subdirectories
        simulation_input_dirs = itertools.cycle(
            filter(lambda p: p.is_dir(), self.simulation_input_dir.glob('*')),
        )

        # Get the first `ensemble_members` directories
        # (i.e., simulation_input_dirs[0:ensemble_members])
        input_dirs: list[Path] = list(
            itertools.islice(simulation_input_dirs, self.ensemble_members),
        )

        # Get the basis states by globbing the input directories
        basis_states = [next(p.glob(self.basis_state_ext)) for p in input_dirs]

        # Return the basis states
        return basis_states

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

    def __init__(self) -> None:
        """Initialize the resampler."""
        # Create a counter to keep track of the simulation IDs
        self.index_counter = itertools.count()

    def get_next_iteration(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Return the simulations for the next iteration."""
        # Reset the index counter
        self.index_counter = itertools.count()

        # Create a list to store the new simulations for this iteration
        simulations = []

        for idx, sim in enumerate(current_iteration):
            # Ensure that the simulation has a restart file, i.e., the `sim`
            # object represents a simulation that has been run.
            assert sim.restart_file is not None

            # Create the metadata for the new simulation
            new_sim = SimulationMetadata(
                weight=sim.weight,
                simulation_id=idx,
                prev_simulation_id=sim.simulation_id,
                restart_file=None,
                parent_restart_file=sim.restart_file,
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
        new_sim = SimulationMetadata(
            weight=weight,
            simulation_id=next(self.index_counter),
            prev_simulation_id=sim.prev_simulation_id,
            restart_file=sim.restart_file,
            parent_restart_file=sim.parent_restart_file,
        )

        # Return the simulation metadata
        return new_sim

    def split_sim(
        self,
        sim: SimulationMetadata,
        n_split: int = 2,
    ) -> list[SimulationMetadata]:
        """Split the parent simulation, `sim`, into `n_split` simulations."""
        # Add the new simulations to the current iteration
        # Use equal weights for the split simulations
        return [
            self._add_new_simulation(sim, sim.weight / n_split)
            for _ in range(n_split)
        ]

    def merge_sims(
        self,
        sims: list[SimulationMetadata],
    ) -> SimulationMetadata:
        """Merge multiple simulations into one."""
        # Get the weights of each simulation to merge
        weights = [sim.weight for sim in sims]

        # Randomly select one of the simulations weighted by their weights
        select: int = np.random.choice(len(sims), p=weights)

        # Add the new simulation to the current iteration
        new_sim = self._add_new_simulation(sims[select], sum(weights))

        # Return the new simulation
        return new_sim

    @abstractmethod
    def resample(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Resample the weighted ensemble."""
        ...


class NaiveResampler(Resampler):
    """Naive resampler."""

    def __init__(
        self,
        pcoord: list[float],
        num_resamples: int = 1,
        n_split: int = 2,
        split_low: bool = True,
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

        """
        self.pcoord = pcoord
        self.num_resamples = num_resamples
        self.n_split = n_split
        self.split_low = split_low

    def split(
        self,
        simulations: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Split the simulation with the lowest progress coordinate."""
        # Find the simulations with the lowest progress coordinate
        sorted_indices = np.argsort(self.pcoord)

        # Split the simulations
        if self.split_low:
            indices = sorted_indices[: self.num_resamples]
        else:
            indices = sorted_indices[-self.num_resamples :]

        # Get the simulations to split
        sims = [simulations[idx] for idx in indices]

        # Split the simulations
        new_sims = []
        for sim in sims:
            new_sims.extend(self.split_sim(sim, self.n_split))

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
            indices = sorted_indices[-self.num_resamples :]
        else:
            indices = sorted_indices[: self.num_resamples]

        # Get the simulations to merge
        sims = [simulations[idx] for idx in indices]

        # Merge the simulations
        new_sim = self.merge_sims(sims)

        return [new_sim]

    def resample(
        self,
        current_iteration: list[SimulationMetadata],
    ) -> list[SimulationMetadata]:
        """Resample the weighted ensemble."""
        # Generate the next iteration
        simulations = self.get_next_iteration(current_iteration)

        # Split the simulations
        to_split = self.split(simulations)

        # Merge the simulations
        to_merge = self.merge(simulations)

        # Add the new simulations to the current iteration
        simulations.extend(to_split)
        simulations.extend(to_merge)

        # Collect any simulations from the previous iteration that were
        # not split or merged
        sims_to_continue = set(simulations)
        sims_to_continue -= set(to_split)
        sims_to_continue -= set(to_merge)

        # Add the simulations to the current iteration
        for sim in sims_to_continue:
            simulations.append(self._add_new_simulation(sim, sim.weight))

        return simulations
