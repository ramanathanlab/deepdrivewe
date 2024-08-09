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

    def recycle_simulations(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        """Recycle the simulations.

        Parameters
        ----------
        cur_sims : list[SimMetadata]
            The list of current simulations.
        next_sims : list[SimMetadata]
            The list of next simulations.

        Returns
        -------
        list[SimMetadata]
            The updated list of current simulations.
        list[SimMetadata]
            The updated list of next simulations.
        """
        # Extract the last progress coordinate from the current simulations
        pcoords = np.array([sim.pcoord[-1] for sim in cur_sims])

        # Get the recycled indices
        recycle_inds = self.recycle(pcoords)

        # Log the recycled indices
        print(f'{recycle_inds=}', flush=True)

        # Create a deep copy of the simulations to prevent modification
        _next_sims = deepcopy(next_sims)
        _cur_sims = deepcopy(cur_sims)

        for idx in recycle_inds:
            # Extract the simulation to recycle
            sim = _next_sims[idx]

            # Choose a random basis state to restart the simulation from
            basis_state = np.random.choice(self.basis_states)

            # Create the metadata for the new simulation
            new_sim = SimMetadata(
                weight=sim.weight,
                simulation_id=sim.simulation_id,
                iteration_id=sim.iteration_id,
                # Set the parent restart file to the basis state
                parent_restart_file=basis_state.parent_restart_file,
                # Set the parent progress coordinate to the basis state
                parent_pcoord=basis_state.parent_pcoord,
                # Set the prev simulation ID to the negative of previous
                # simulation to indicate that the simulation is recycled.
                # Add 1 to the simulation ID to avoid negative zero.
                parent_simulation_id=-(sim.simulation_id + 1),
                # TODO: Can we double check this is correct?
                wtg_parent_ids=sim.wtg_parent_ids,
            )

            # Log the recycled simulation
            print(f'Recycling simulation {new_sim}', flush=True)

            # Add the new simulation to the current iteration
            _next_sims[idx] = new_sim

            # Update the endpoint_type of the current simulation
            # to indicate that it was recycled
            _cur_sims[idx].endpoint_type = 3

        return _cur_sims, _next_sims

    @abstractmethod
    def recycle(self, pcoords: np.ndarray) -> np.ndarray:
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

    def recycle(self, pcoords: np.ndarray) -> np.ndarray:
        """Recycle the simulations under the target threshold.

        Parameters
        ----------
        pcoords : np.ndarray
            The progress coordinates for the simulations.
            Shape: (n_simulations, n_dims).

        Returns
        -------
        np.ndarray
            The list of simulation indices to recycle. Shape: (n_recycled,)
        """
        return np.where(pcoords[:, self.pcoord_idx] < self.target_threshold)[0]


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

    def recycle(self, pcoords: np.ndarray) -> np.ndarray:
        """Recycle the simulations above the target threshold.

        Parameters
        ----------
        pcoords : np.ndarray
            The progress coordinates for the simulations.
            Shape: (n_simulations, n_dims).

        Returns
        -------
        np.ndarray
            The list of simulation indices to recycle. Shape: (n_recycled,)
        """
        return np.where(pcoords[:, self.pcoord_idx] > self.target_threshold)[0]
