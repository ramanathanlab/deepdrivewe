from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
from westpa_colmena.apps.amber_simulation import SimulationResult
from dataclasses import dataclass, field

# TODO: It would be nice to support merge multiple into 1 and split 1 into multiple

def westpa_logic() -> None:
    """Analyze the current batch of simulations and output a new set of starting points."""
    # westpa_split()
    # westpa_merge()


@dataclass
class SimulationMetadata:
    weight: float
    simulation_id: int
    prev_simulation_id: int | None = field(
        default=None,
        metadata={
            "help": "The ID of the previous simulation the current one is split from, or None if it's a basis state."
        },
    )
    restart_file : Path | None = field(
        default=None,
        metadata={
            "help": "The restart file for the simulation."
        },
    )
    # TODO: This may not need to be None because the bstates can populate it
    parent_restart_file: Path | None = field(
        default=None,
        metadata={
            "help": "The restart file for the parent simulation."
        },
    )

    def __hash__(self) -> int:
        """Hash the simulation metadata to ensure that it is unique."""
        return hash((self.simulation_id, self.restart_file))


class WeightedEnsemble:
    # TODO: Figure out a checkpointing mechanism for the metadata

    # The list of simulations for each iteration
    simulations: list[list[SimulationMetadata]]

    # The current iteration of the weighted ensemble
    iteration_idx: int

    def __init__(self, basis_states: list[Path], ensemble_members: int) -> None:
        # The current iteration of the weighted ensemble
        self.iteration_idx = 0
        # The list of simulations for each iteration
        self.simulations = []
        # Assign a uniform weight to each of the basis states
        weight = 1.0 / ensemble_members

        # Create a generator that will cycle through the basis states (e.g.,
        # if ensemble_members is 3 and there are 2 basis states, the generator
        # will yield [0, basis_state1], [1, basis_state2], [2, basis_state1])
        sim_generator = zip(range(ensemble_members), itertools.cycle(basis_states))

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

    def _add_new_simulation(self, sim: SimulationMetadata, weight: float) -> None:
        """Add a new simulation to the current iteration."""
        new_simulation = SimulationMetadata(
                weight=weight,
                simulation_id=len(self.current_iteration),
                prev_simulation_id=sim.simulation_id,
                restart_file=None,
                parent_restart_file=sim.restart_file
        )
        self.current_iteration.append(new_simulation)

    def advance_iteration(
        self,
        to_split: list[SimulationMetadata],
        to_merge: list[list[SimulationMetadata]],
        n_split: int = 2,
    ) -> None:
        """Advance the iteration of the weighted ensemble."""
        # Create a list to store the new simulations for this iteration
        self.simulations.append([])
        self.iteration_idx += 1

        # Split the simulations
        for sim in to_split:
            self._split(sim, n_split=n_split)

        # Merge the simulations
        for sims in to_merge:
            self._merge(sims)

        # Collect any simulations from the previous iteration that were not split or merged
        sims_to_continue = set(self.simulations[self.iteration_idx - 1])
        sims_to_continue -= set(to_split)
        sims_to_continue -= set(itertools.chain(*to_merge))

        # Add the simulations to the current iteration
        for sim in sims_to_continue:
            self._add_new_simulation(sim, sim.weight)

    def _split(self, sim: SimulationMetadata, n_split: int = 2) -> None:
        """Split the parent simulation, `sim`, into `n_split` simulations with equal weight."""

        # Add the new simulations to the current iteration
        for _ in range(n_split):
            self._add_new_simulation(sim, sim.weight / n_split)

    def _merge(self, sims: list[SimulationMetadata]) -> None:
        """Merge multiple simulations into one."""

        # Get the weights of each simulation to merge
        weights = [sim.weight for sim in sims]

        # Randomly select one of the simulations weighted by their weights
        select: int = np.random.choice(len(sims), p=weights)

        # Add the new simulation to the current iteration
        self._add_new_simulation(sims[select], sum(weights))


def run_inference(input_data: list[SimulationResult]) -> None:
    pcoords = [item.pcoord for item in input_data]

    # Pick the simulation indices to restart
    num_simulations = len(input_data)
    restart_simulations = np.random.randint(
        low=0, high=num_simulations, size=num_simulations
    )

    # Create weights for each of the simulations
