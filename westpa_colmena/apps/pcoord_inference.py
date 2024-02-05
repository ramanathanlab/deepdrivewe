from __future__ import annotations
from pathlib import Path
import numpy as np
from westpa_colmena.apps.amber_simulation import SimulationResult
import numpy as np
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


# TODO: Pick up working from here, we know the simulation metadata when
# splitting, and we can update the simulation result when the simulation returns
@dataclass
class SimulationData:
    simulation: SimulationMetadata
    result: SimulationResult


class WeightedEnsemble:
    data: list[SimulationResult]
    simulations: list[SimulationMetadata]

    def __init__(self, basis_states: list[SimulationResult]) -> None:
        self.data = basis_states
        # Assign a uniform weight to each of the basis states
        n_states = len(basis_states)
        weight = 1.0 / n_states
        self.simulations = [
            SimulationMetadata(
                weight=weight, simulation_id=idx, prev_simulation_id=None
            )
            for idx in range(n_states)
        ]

    def get_parent_restart(self, sim: SimulationMetadata) -> Path | None:
        """Get the restart file for the parent simulation."""
        if sim.prev_simulation_id is None:
            return None
        return self.data[sim.prev_simulation_id].restart_file

    # TODO: There is a bug since we currently don't have a way to get the
    # simulation result for the new simulation
    def split(self, sim: SimulationMetadata, result: SimulationResult) -> None:
        """Split the parent simulation, `sim`, into two simulations where
        `result` is the data from the new simulation."""

        # Create a new metadata object to represent the new simulation info
        new_simulation = SimulationMetadata(
            weight=sim.weight / 2,
            simulation_id=len(self.simulations),
            prev_simulation_id=sim.simulation_id,
        )

        # TODO: Probably easier to store the restart_file and parent_restart_file
        # in the SimulationMetadata object so that we don't have to store everything
        # (pcoords, coords, etc) in the SimulationResult object. This means we also
        # don't need to maintain a separate list of SimulationResult objects which will
        # make it easier to checkpoint the WeightedEnsemble object, since it will only
        # store be plain old data and otherwise be stateless. We could store it as a json
        # file or something similar once per iteration.

        # Update the weight of the parent simulation so that the total
        # weight of the ensemble sums to one
        self.simulations[sim.simulation_id].weight /= 2

        # Setup a new simulation result object for the new simulation
        assert result.restart_file is not None
        new_result = SimulationResult(
            pcoord=[],
            coords=[],
            restart_file=None,
            parent_restart_file=result.restart_file
        )

        # Add the new simulation data to the lists
        self.data.append(new_result)
        self.simulations.append(new_simulation)

    def merge(
        self,
        sim1: SimulationMetadata,
        sim2: SimulationMetadata,
        result1: SimulationResult,
        result2: SimulationResult,
    ) -> None:
        """Merge two simulations into one."""

        # TODO: Consider making dataclass to collect the simulation data and metadata

        # Randomly select one of the two simulations weighted by their weights
        select = np.random.choice([0, 1], p=[sim1.weight, sim2.weight])
        sim = [sim1, sim2][select]
        result = [result1, result2][select]

        # Create a new metadata object to represent the new simulation info
        new_simulation = SimulationMetadata(
            weight=sim1.weight + sim2.weight,
            simulation_id=len(self.simulations),
            prev_simulation_id=sim.simulation_id,
        )

        # Add the new simulation data to the lists
        self.data.append(result)
        self.simulations.append(new_simulation)


def run_inference(input_data: list[SimulationResult]) -> None:
    pcoords = [item.pcoord for item in input_data]

    # Pick the simulation indices to restart
    num_simulations = len(input_data)
    restart_simulations = np.random.randint(
        low=0, high=num_simulations, size=num_simulations
    )

    # Create weights for each of the simulations
