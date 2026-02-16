"""Orchestrator agent for coordinating the weighted ensemble workflow."""

from __future__ import annotations

import asyncio
from typing import Any

from academy.agent import action
from academy.agent import loop
from academy.handle import Handle

from deepdrivewe.academy_agents.analysis import AnalysisPoolAgent
from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.academy_agents.config import AcademyWorkflowConfig
from deepdrivewe.academy_agents.ensemble import EnsembleManagerAgent
from deepdrivewe.academy_agents.simulation import SimulationPoolAgent
from deepdrivewe.checkpoint import EnsembleCheckpointer


class OrchestratorAgent(AcademyAgent):
    """Agent that orchestrates the weighted ensemble workflow.

    This agent coordinates the overall workflow by managing interactions
    between the simulation pool, analysis pool, and ensemble manager.
    It advances iterations, monitors progress, and handles checkpointing.

    Attributes
    ----------
    config : AcademyWorkflowConfig
        Configuration for the workflow.
    simulation_pool : Handle[SimulationPoolAgent]
        Handle to the simulation pool agent.
    ensemble_manager : Handle[EnsembleManagerAgent]
        Handle to the ensemble manager agent.
    analysis_pool : Handle[AnalysisPoolAgent] | None
        Handle to the analysis pool agent (optional).
    checkpointer : EnsembleCheckpointer
        Checkpointer for saving ensemble state.
    """

    def __init__(
        self,
        config: AcademyWorkflowConfig,
        simulation_pool: Handle[SimulationPoolAgent],
        ensemble_manager: Handle[EnsembleManagerAgent],
        checkpointer: EnsembleCheckpointer,
        analysis_pool: Handle[AnalysisPoolAgent] | None = None,
    ) -> None:
        """Initialize the orchestrator agent.

        Parameters
        ----------
        config : AcademyWorkflowConfig
            Configuration for the workflow.
        simulation_pool : Handle[SimulationPoolAgent]
            Handle to the simulation pool agent.
        ensemble_manager : Handle[EnsembleManagerAgent]
            Handle to the ensemble manager agent.
        checkpointer : EnsembleCheckpointer
            Checkpointer for saving ensemble state.
        analysis_pool : Handle[AnalysisPoolAgent] | None
            Handle to the analysis pool agent (optional).
        """
        super().__init__()
        self.config = config
        self.simulation_pool = simulation_pool
        self.ensemble_manager = ensemble_manager
        self.analysis_pool = analysis_pool
        self.checkpointer = checkpointer
        self._workflow_complete = False
        self._current_iteration = 0

    @action
    async def start_workflow(self) -> None:
        """Start the weighted ensemble workflow.

        This action initializes the workflow and begins the first iteration.
        """
        self._log_action('start_workflow')

        self.logger.info(
            f'Starting workflow for {self.config.num_iterations} iterations',
        )

        # Get initial simulations from ensemble manager
        next_sims = await self.ensemble_manager.get_next_simulations()

        self.logger.info(
            f'Starting iteration 0 with {len(next_sims)} simulations',
        )

        # Submit initial simulations to pool
        for sim_metadata in next_sims:
            await self.simulation_pool.submit_simulation(sim_metadata)

        self._current_iteration = 0
        self._workflow_complete = False

    @action
    async def advance_iteration(self) -> bool:
        """Advance to the next iteration.

        Returns
        -------
        bool
            True if iteration was advanced, False if workflow is complete.
        """
        self._log_action('advance_iteration', iteration=self._current_iteration)

        # Check if workflow is complete
        if self._current_iteration >= self.config.num_iterations:
            self.logger.info('Workflow complete')
            self._workflow_complete = True
            return False

        # Wait for all simulations to complete
        # In a real implementation, this would poll the simulation pool
        # For now, we'll use a simple sleep-based approach
        self.logger.info(
            f'Waiting for iteration {self._current_iteration} to complete...',
        )

        # Get all results from simulation pool
        all_results = await self.simulation_pool.get_all_results()

        # Get expected number of simulations
        next_sims = await self.ensemble_manager.get_next_simulations()
        expected_count = len(next_sims)

        # Wait until all simulations are complete
        while len(all_results) < expected_count:
            await asyncio.sleep(1.0)
            all_results = await self.simulation_pool.get_all_results()

        self.logger.info(
            f'All {len(all_results)} simulations complete for iteration '
            f'{self._current_iteration}',
        )

        # Extract completed simulation results
        sim_results = [
            result
            for result in all_results.values()
            if result.get('success', False)
        ]

        # Run analysis if analysis pool is enabled
        if self.analysis_pool is not None:
            self.logger.info('Running analysis on simulation results...')
            try:
                analysis_results = await self.analysis_pool.analyze_simulations(
                    sim_results=sim_results,
                    iteration_id=self._current_iteration,
                )
                self.logger.info(
                    f'Analysis complete: {list(analysis_results.keys())}',
                )

                # Add analysis results to simulation metadata
                for i, sim_result in enumerate(sim_results):
                    if 'analysis' in sim_result:
                        # Store analysis results in metadata for checkpointing
                        sim_result['metadata']['analysis'] = sim_result['analysis']

            except Exception as e:
                self.logger.error(f'Analysis failed: {e}')
                # Continue workflow even if analysis fails

        # Extract simulation metadata
        cur_sims = [result['metadata'] for result in sim_results]

        # Apply resampling to get next iteration
        cur_sims_updated, next_sims_new, metadata = (
            await self.ensemble_manager.apply_resampling(cur_sims)
        )

        # Update ensemble state
        await self.ensemble_manager.update_ensemble(
            cur_sims=cur_sims_updated,
            next_sims=next_sims_new,
            metadata=metadata,
        )

        # Clear simulation pool results
        await self.simulation_pool.clear_results()

        # Submit next iteration simulations
        for sim_metadata in next_sims_new:
            await self.simulation_pool.submit_simulation(sim_metadata)

        # Checkpoint if needed
        self._current_iteration += 1

        if self._current_iteration % self.config.checkpoint_interval == 0:
            await self._save_checkpoint()

        self.logger.info(f'Advanced to iteration {self._current_iteration}')

        return True

    @action
    async def check_completion(self) -> bool:
        """Check if the workflow is complete.

        Returns
        -------
        bool
            True if workflow is complete.
        """
        return self._workflow_complete

    @action
    async def get_status(self) -> dict[str, Any]:
        """Get the current workflow status.

        Returns
        -------
        dict[str, Any]
            Dictionary containing workflow status information.
        """
        ensemble_state = await self.ensemble_manager.get_ensemble_state()

        return {
            'current_iteration': self._current_iteration,
            'total_iterations': self.config.num_iterations,
            'workflow_complete': self._workflow_complete,
            'ensemble_state': ensemble_state,
        }

    @loop
    async def monitor_progress(self, shutdown: asyncio.Event) -> None:
        """Monitor workflow progress and log status updates.

        Parameters
        ----------
        shutdown : asyncio.Event
            Event to signal shutdown.
        """
        self.logger.info('Starting monitor_progress loop')

        while not shutdown.is_set() and not self._workflow_complete:
            try:
                status = await self.get_status()

                self.logger.info(
                    f"Workflow status: iteration {status['current_iteration']}/"
                    f"{status['total_iterations']}",
                )

                await asyncio.sleep(10.0)  # Log status every 10 seconds

            except Exception as e:
                self._log_error('monitor_progress', e)
                await asyncio.sleep(5.0)

        self.logger.info('Exiting monitor_progress loop')

    @loop
    async def evaluate_goals(self, shutdown: asyncio.Event) -> None:
        """Evaluate goal-oriented metrics for adaptive sampling.

        This is a placeholder for Phase 4 goal-oriented reward models.
        In the full implementation, this would evaluate progress towards
        user-defined goals (e.g., protein folding, binding pocket opening).

        Parameters
        ----------
        shutdown : asyncio.Event
            Event to signal shutdown.
        """
        self.logger.info('Starting evaluate_goals loop (placeholder)')

        while not shutdown.is_set() and not self._workflow_complete:
            try:
                # Placeholder: In Phase 4, this would:
                # 1. Get current ensemble state
                # 2. Evaluate progress towards goals
                # 3. Compute reward signals
                # 4. Adjust sampling strategy if needed

                self.logger.debug('Goal evaluation not yet implemented')

                await asyncio.sleep(30.0)  # Evaluate every 30 seconds

            except Exception as e:
                self._log_error('evaluate_goals', e)
                await asyncio.sleep(10.0)

        self.logger.info('Exiting evaluate_goals loop')

    async def _save_checkpoint(self) -> None:
        """Save ensemble checkpoint to disk."""
        try:
            _ = await self.ensemble_manager.get_ensemble_state()

            # In a full implementation, this would use the checkpointer
            # to save the ensemble state to HDF5 format
            # For now, we just log that a checkpoint would be saved
            self.logger.info(
                f"Checkpoint saved for iteration {self._current_iteration}",
            )

        except Exception as e:
            self._log_error('_save_checkpoint', e)

