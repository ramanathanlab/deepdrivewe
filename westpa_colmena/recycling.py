"""Recycling algorithms for the weighted ensemble."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy

import numpy as np

from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import SimMetadata


class Recycler(ABC):
    """Recycler for the weighted ensemble."""

    def __init__(self, basis_states: BasisStates) -> None:
        """Initialize the recycler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        """
        self.basis_states = basis_states

    def get_pcoords(
        self,
        sims: list[SimMetadata],
        pcoord_idx: int = 0,
    ) -> list[float]:
        """Extract the progress coordinate from the simulations.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulation metadata.
        pcoord_idx : int
            The index of the progress coordinate to extract. Default is 0.

        Returns
        -------
        list[float]
            The progress coordinates for the simulations.
        """
        # Extract the progress coordinates
        pcoords = []
        for sim in sims:
            # Ensure that the simulation has a progress coordinate
            assert sim.pcoord is not None
            # We only extract the progress coordinate at the specified index
            pcoords.append(sim.pcoord[pcoord_idx])
        return pcoords

    def recycle_simulations(
        self,
        sims: list[SimMetadata],
        recycle_indices: list[int],
    ) -> list[SimMetadata]:
        """Recycle the simulations.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations to recycle.

        Returns
        -------
        list[SimMetadata]
            The list of recycled simulations.
        """
        # Log the recycled indices
        print(f'{recycle_indices=}', flush=True)

        # Create a deep copy of the simulations to prevent modification
        recycled_sims = deepcopy(sims)

        for idx in recycle_indices:
            sim = recycled_sims[idx]

            # Choose a random basis state to restart the simulation from
            basis_state = np.random.choice(self.basis_states)

            # Create the metadata for the new simulation
            new_sim = SimMetadata(
                weight=sim.weight,
                simulation_id=idx,
                iteration_id=sim.iteration_id + 1,
                # Set the prev simulation ID to the negative of previous
                # simulation to indicate that the simulation is recycled
                parent_simulation_id=sim.simulation_id * -1,
                # Set the parent restart file to the basis state
                parent_restart_file=basis_state.parent_restart_file,
                # Set the parent progress coordinate to the basis state
                parent_pcoord=basis_state.parent_pcoord,
            )

            # Log the recycled simulation
            print(f'Recycling simulation {new_sim}', flush=True)

            # Add the new simulation to the current iteration
            recycled_sims[idx] = new_sim

        return recycled_sims

    @abstractmethod
    def recycle(self, sims: list[SimMetadata]) -> list[SimMetadata]:
        """Return a list of simulation indices to recycle."""
        ...


class LowRecycler(Recycler):
    """Recylcle simulations under a certain progress coordinate threshold."""

    def __init__(
        self,
        basis_states: BasisStates,
        target_threshold: float,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the recycler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state.
        pcoord_idx : int
            The index of the progress coordinate to use for recycling. Only
            applicable if a multi-dimensional pcoord is used, will choose the
            specified index of the pcoord for recycling. Default is 0.
        """
        super().__init__(basis_states)
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def recycle(self, sims: list[SimMetadata]) -> list[SimMetadata]:
        """Recycle the simulations under the target threshold.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations to recycle.

        Returns
        -------
        list[SimMetadata]
            The list of recycled simulations.
        """
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Recycle the simulations
        recycle_indices = [
            i for i, p in enumerate(pcoords) if p < self.target_threshold
        ]

        return self.recycle_simulations(sims, recycle_indices)


class HighRecycler(Recycler):
    """Recylcle simulations above a certain progress coordinate threshold."""

    def __init__(
        self,
        basis_states: BasisStates,
        target_threshold: float,
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the recycler.

        Parameters
        ----------
        basis_states : BasisStates
            The basis states for the weighted ensemble.
        target_threshold : float
            The target threshold for the progress coordinate to be considered
            in the target state.
        pcoord_idx : int
            The index of the progress coordinate to use for recycling. Only
            applicable if a multi-dimensional pcoord is used, will choose the
            specified index of the pcoord for recycling. Default is 0.
        """
        super().__init__(basis_states)
        self.target_threshold = target_threshold
        self.pcoord_idx = pcoord_idx

    def recycle(self, sims: list[SimMetadata]) -> list[SimMetadata]:
        """Recycle the simulations above the target threshold.

        Parameters
        ----------
        sims : list[SimMetadata]
            The list of simulations to recycle.

        Returns
        -------
        list[SimMetadata]
            The list of recycled simulations.
        """
        # Extract the progress coordinates
        pcoords = self.get_pcoords(sims, self.pcoord_idx)

        # Recycle the simulations
        recycle_indices = [
            i for i, p in enumerate(pcoords) if p > self.target_threshold
        ]

        return self.recycle_simulations(sims, recycle_indices)
