"""Ensemble manager agent for weighted ensemble state management."""

from __future__ import annotations

from typing import Any

from academy.agent import action

from deepdrivewe import IterationMetadata
from deepdrivewe import SimMetadata
from deepdrivewe import WeightedEnsemble
from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.binners.base import Binner
from deepdrivewe.recyclers.base import Recycler
from deepdrivewe.resamplers.base import Resampler


class EnsembleManagerAgent(AcademyAgent):
    """Agent that manages weighted ensemble state and resampling.

    This agent wraps the existing WeightedEnsemble logic and provides
    Academy actions for binning, resampling, and recycling simulations.
    It maintains the ensemble state across iterations and coordinates
    with the orchestrator to advance the workflow.

    The EnsembleManagerAgent is stateful and maintains:
    - Current weighted ensemble state
    - Binning, resampling, and recycling policies
    - Iteration metadata

    Attributes
    ----------
    ensemble : WeightedEnsemble
        The weighted ensemble being managed.
    binner : Binner
        Binner for assigning simulations to bins.
    resampler : Resampler
        Resampler for splitting/merging simulations.
    recycler : Recycler
        Recycler for handling failed simulations.
    """

    def __init__(
        self,
        ensemble: WeightedEnsemble,
        binner: Binner,
        resampler: Resampler,
        recycler: Recycler,
    ) -> None:
        """Initialize the ensemble manager agent.

        Parameters
        ----------
        ensemble : WeightedEnsemble
            The weighted ensemble to manage.
        binner : Binner
            Binner for assigning simulations to bins.
        resampler : Resampler
            Resampler for splitting/merging simulations.
        recycler : Recycler
            Recycler for handling failed simulations.
        """
        super().__init__()
        self.ensemble = ensemble
        self.binner = binner
        self.resampler = resampler
        self.recycler = recycler

    @action
    async def get_next_simulations(self) -> list[dict[str, Any]]:
        """Get the next simulations to run.

        Returns
        -------
        list[dict[str, Any]]
            List of simulation metadata dictionaries for the next iteration.
        """
        self._log_action('get_next_simulations')

        # Convert SimMetadata objects to dictionaries for serialization
        next_sims = [sim.model_dump() for sim in self.ensemble.next_sims]

        self.logger.info(
            f'Returning {len(next_sims)} simulations for iteration '
            f'{self.ensemble.iteration}',
        )

        return next_sims

    @action
    async def update_ensemble(
        self,
        cur_sims: list[dict[str, Any]],
        next_sims: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Update the ensemble with completed iteration results.

        Parameters
        ----------
        cur_sims : list[dict[str, Any]]
            Completed simulations from the current iteration.
        next_sims : list[dict[str, Any]]
            Simulations for the next iteration.
        metadata : dict[str, Any]
            Iteration metadata.
        """
        self._log_action('update_ensemble', iteration=metadata.get('iteration_id'))

        # Convert dictionaries back to Pydantic models
        cur_sims_objs = [SimMetadata(**sim) for sim in cur_sims]
        next_sims_objs = [SimMetadata(**sim) for sim in next_sims]
        metadata_obj = IterationMetadata(**metadata)

        # Advance the ensemble iteration
        self.ensemble.advance_iteration(
            cur_sims=cur_sims_objs,
            next_sims=next_sims_objs,
            metadata=metadata_obj,
        )

        self.logger.info(
            f'Advanced ensemble to iteration {self.ensemble.iteration}',
        )

    @action
    async def apply_binning(
        self,
        sims: list[dict[str, Any]],
    ) -> dict[int, list[int]]:
        """Assign simulations to bins.

        Parameters
        ----------
        sims : list[dict[str, Any]]
            Simulations to bin.

        Returns
        -------
        dict[int, list[int]]
            Bin assignments mapping bin index to simulation indices.
        """
        self._log_action('apply_binning', num_sims=len(sims))

        # Convert to SimMetadata objects
        sim_objs = [SimMetadata(**sim) for sim in sims]

        # Apply binning
        bin_assignments = self.binner.bin_simulations(sim_objs)

        self.logger.info(f'Assigned {len(sims)} sims to {len(bin_assignments)} bins')

        return bin_assignments

    @action
    async def apply_resampling(
        self,
        cur_sims: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Apply resampling to the current simulations.

        This action runs the full resampling pipeline including binning,
        recycling, and resampling to produce the next iteration of simulations.

        Parameters
        ----------
        cur_sims : list[dict[str, Any]]
            Current simulations to resample.

        Returns
        -------
        tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
            Tuple of (current_sims, next_sims, metadata) after resampling.
        """
        self._log_action('apply_resampling', num_sims=len(cur_sims))

        # Convert to SimMetadata objects
        cur_sims_objs = [SimMetadata(**sim) for sim in cur_sims]

        # Run the resampling pipeline
        try:
            cur_sims_result, next_sims_result, metadata = self.resampler.run(
                cur_sims=cur_sims_objs,
                binner=self.binner,
                recycler=self.recycler,
            )

            self.logger.info(
                f'Resampling produced {len(next_sims_result)} simulations '
                f'for next iteration',
            )

            # Convert back to dictionaries
            return (
                [sim.model_dump() for sim in cur_sims_result],
                [sim.model_dump() for sim in next_sims_result],
                metadata.model_dump(),
            )

        except Exception as e:
            self._log_error('apply_resampling', e, num_sims=len(cur_sims))
            raise

    @action
    async def apply_recycling(
        self,
        cur_sims: list[dict[str, Any]],
        next_sims: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply recycling to failed simulations.

        Parameters
        ----------
        cur_sims : list[dict[str, Any]]
            Current simulations.
        next_sims : list[dict[str, Any]]
            Next simulations to potentially recycle.

        Returns
        -------
        tuple[list[dict[str, Any]], list[dict[str, Any]]]
            Updated (current_sims, next_sims) after recycling.
        """
        self._log_action('apply_recycling', num_sims=len(next_sims))

        # Convert to SimMetadata objects
        cur_sims_objs = [SimMetadata(**sim) for sim in cur_sims]
        next_sims_objs = [SimMetadata(**sim) for sim in next_sims]

        # Apply recycling
        cur_sims_result, next_sims_result = self.recycler.recycle_simulations(
            cur_sims=cur_sims_objs,
            next_sims=next_sims_objs,
        )

        self.logger.info('Recycling complete')

        # Convert back to dictionaries
        return (
            [sim.model_dump() for sim in cur_sims_result],
            [sim.model_dump() for sim in next_sims_result],
        )

    @action
    async def get_current_iteration(self) -> int:
        """Get the current iteration number.

        Returns
        -------
        int
            Current iteration number.
        """
        return self.ensemble.iteration

    @action
    async def get_ensemble_state(self) -> dict[str, Any]:
        """Get the current ensemble state.

        Returns
        -------
        dict[str, Any]
            Dictionary containing ensemble state information.
        """
        return {
            'iteration': self.ensemble.iteration,
            'num_current_sims': len(self.ensemble.cur_sims),
            'num_next_sims': len(self.ensemble.next_sims),
            'metadata': self.ensemble.metadata.model_dump(),
        }

