"""Weighted ensemble logic."""

from __future__ import annotations

import itertools
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

    # The current iteration of the weighted ensemble
    iteration_idx: int

    def __init__(
        self,
        basis_states: list[Path],
        ensemble_members: int,
    ) -> None:
        """Initialize the weighted ensemble.

        Parameters
        ----------
        basis_states : list[Path]
            The basis states for the weighted ensemble.
        ensemble_members : int
            The number of simulations to start the weighted ensemble with.
        """
        # The current iteration of the weighted ensemble
        self.iteration_idx = 0
        # The list of simulations for each iteration
        self.simulations = []
        # Assign a uniform weight to each of the basis states
        weight = 1.0 / ensemble_members

        # Create a generator that will cycle through the basis states (e.g.,
        # if ensemble_members is 3 and there are 2 basis states, the generator
        # will yield [0, basis_state1], [1, basis_state2], [2, basis_state1])
        sim_generator = zip(
            range(ensemble_members),
            itertools.cycle(basis_states),
        )

        # Create the metadata for each basis state
        sims = [
            SimulationMetadata(
                weight=weight,
                simulation_id=idx,
                prev_simulation_id=None,
                restart_file=None,
                parent_restart_file=basis_state,
            )
            for idx, basis_state in sim_generator
        ]

        self.simulations.append(sims)

    @property
    def current_iteration(self) -> list[SimulationMetadata]:
        """Return the simulations for the current iteration."""
        return self.simulations[self.iteration_idx]

    def _add_new_simulation(
        self,
        sim: SimulationMetadata,
        weight: float,
    ) -> None:
        """Add a new simulation to the current iteration."""
        # Ensure that the simulation has a restart file, i.e., the `sim`
        # object represents a simulation that has been run.
        assert sim.restart_file is not None

        # Create the metadata for the new simulation
        new_simulation = SimulationMetadata(
            weight=weight,
            simulation_id=len(self.current_iteration),
            prev_simulation_id=sim.simulation_id,
            restart_file=None,
            parent_restart_file=sim.restart_file,
        )

        # Add the new simulation to the current iteration
        self.current_iteration.append(new_simulation)

    def advance_iteration(
        self,
        to_split: list[SimulationMetadata],
        to_merge: list[list[SimulationMetadata]],
        n_split: int = 2,
    ) -> None:
        """Advance the iteration of the weighted ensemble.

        The binner is responsible for determining which simulations to split
        and merge. The binner will then call this method to advance the
        iteration of the weighted ensemble.
        """
        # Create a list to store the new simulations for this iteration
        self.simulations.append([])
        self.iteration_idx += 1

        # Split the simulations
        for sim in to_split:
            self._split(sim, n_split=n_split)

        # Merge the simulations
        for sims in to_merge:
            self._merge(sims)

        # Collect any simulations from the previous iteration that were
        # not split or merged
        sims_to_continue = set(self.simulations[self.iteration_idx - 1])
        sims_to_continue -= set(to_split)
        sims_to_continue -= set(itertools.chain(*to_merge))

        # Add the simulations to the current iteration
        for sim in sims_to_continue:
            self._add_new_simulation(sim, sim.weight)

    def _split(self, sim: SimulationMetadata, n_split: int = 2) -> None:
        """Split the parent simulation, `sim`, into `n_split` simulations."""
        # Add the new simulations to the current iteration
        for _ in range(n_split):
            # Use equal weights for the split simulations
            self._add_new_simulation(sim, sim.weight / n_split)

    def _merge(self, sims: list[SimulationMetadata]) -> None:
        """Merge multiple simulations into one."""
        # Get the weights of each simulation to merge
        weights = [sim.weight for sim in sims]

        # Randomly select one of the simulations weighted by their weights
        select: int = np.random.choice(len(sims), p=weights)

        # Add the new simulation to the current iteration
        self._add_new_simulation(sims[select], sum(weights))
