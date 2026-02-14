"""Simulation agents for running MD simulations."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

from academy.agent import action
from academy.agent import loop
from academy.handle import Handle

from deepdrivewe import SimMetadata
from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.academy_agents.config import SimulationPoolConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation


class SimulationAgent(AcademyAgent):
    """Agent that executes individual MD simulations.

    This agent runs OpenMM simulations and returns trajectory data.
    It maintains a queue of simulation tasks and processes them
    sequentially in its await_task loop.

    Attributes
    ----------
    config : SimulationPoolConfig
        Configuration for simulations.
    current_task : dict[str, Any] | None
        Currently executing simulation task.
    is_busy : bool
        Whether the agent is currently running a simulation.
    """

    def __init__(self, config: SimulationPoolConfig) -> None:
        """Initialize the simulation agent.

        Parameters
        ----------
        config : SimulationPoolConfig
            Configuration for simulations.
        """
        super().__init__()
        self.config = config
        self.current_task: dict[str, Any] | None = None
        self.is_busy = False
        self._task_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

    @action
    async def run_simulation(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run an MD simulation.

        Parameters
        ----------
        metadata : dict[str, Any]
            Simulation metadata dictionary.

        Returns
        -------
        dict[str, Any]
            Simulation result containing trajectory data and updated metadata.
        """
        self._log_action('run_simulation', sim_id=metadata.get('simulation_id'))

        # Convert to SimMetadata object
        sim_metadata = SimMetadata(**metadata)

        # Mark simulation start
        sim_metadata.mark_simulation_start()

        try:
            # Create simulation output directory
            sim_output_dir = (
                self.config.output_dir / sim_metadata.simulation_name
            )

            # Remove directory if it exists (from previous failed attempt)
            if sim_output_dir.exists():
                await asyncio.sleep(1)  # Avoid NFS race conditions
                shutil.rmtree(sim_output_dir)

            sim_output_dir.mkdir(parents=True, exist_ok=True)

            # Log the config to the output directory
            self.config.simulation_config.dump_yaml(
                sim_output_dir / 'config.yaml',
            )

            # Initialize OpenMM simulation
            simulation = OpenMMSimulation(
                config=self.config.simulation_config,
                output_dir=sim_output_dir,
                checkpoint_file=sim_metadata.parent_restart_file,
            )

            # Create RMSD reporter for progress coordinate calculation
            from deepdrivewe.simulation.openmm import ContactMapRMSDReporter

            reporter = ContactMapRMSDReporter(
                report_interval=self.config.simulation_config.report_steps,
                reference_file=self.config.reference_file,
                cutoff_angstrom=self.config.cutoff_angstrom,
                mda_selection=self.config.mda_selection,
                openmm_selection=self.config.openmm_selection,
            )

            # Run the simulation (blocking operation)
            # We run this in a thread pool to avoid blocking the event loop
            await asyncio.to_thread(simulation.run, reporters=[reporter])

            # Extract progress coordinate (RMSD values)
            pcoord = reporter.get_rmsds()

            # Get trajectory data
            trajectory_data = {
                'restart_file': str(simulation.restart_file),
                'trajectory_file': str(simulation.trajectory_file),
                'log_file': str(simulation.log_file),
            }

            # Update metadata with progress coordinate
            sim_metadata.restart_file = simulation.restart_file
            sim_metadata.pcoord = pcoord.tolist()
            sim_metadata.mark_simulation_end()

            self.logger.info(
                f'Completed simulation {sim_metadata.simulation_id} '
                f'in {sim_metadata.walltime:.2f}s',
            )

            return {
                'metadata': sim_metadata.model_dump(),
                'trajectory': trajectory_data,
                'success': True,
            }

        except Exception as e:
            self._log_error('run_simulation', e, sim_id=metadata.get('simulation_id'))
            sim_metadata.mark_simulation_end()

            return {
                'metadata': sim_metadata.model_dump(),
                'trajectory': {},
                'success': False,
                'error': str(e),
            }

    @action
    async def is_available(self) -> bool:
        """Check if the agent is available for work.

        Returns
        -------
        bool
            True if the agent is not busy.
        """
        return not self.is_busy

    @action
    async def enqueue_task(self, metadata: dict[str, Any]) -> None:
        """Add a simulation task to the queue.

        Parameters
        ----------
        metadata : dict[str, Any]
            Simulation metadata.
        """
        await self._task_queue.put(metadata)
        self.logger.debug(f'Enqueued task {metadata.get("simulation_id")}')

    @action
    async def get_trajectory(self) -> dict[str, Any]:
        """Get trajectory data from the most recent simulation.

        Returns
        -------
        dict[str, Any]
            Trajectory data including file paths and coordinates.
        """
        if self.current_task is None:
            return {}

        return self.current_task.get('trajectory', {})

    @action
    async def checkpoint(self) -> dict[str, Any]:
        """Save checkpoint of current simulation state.

        Returns
        -------
        dict[str, Any]
            Checkpoint information.
        """
        checkpoint_data = {
            'is_busy': self.is_busy,
            'current_task': self.current_task,
            'queue_size': self._task_queue.qsize(),
        }

        self.logger.debug(f'Checkpoint: {checkpoint_data}')
        return checkpoint_data

    @loop
    async def await_task(self, shutdown: asyncio.Event) -> None:
        """Process queued simulation tasks.

        This loop continuously processes tasks from the queue until
        the shutdown event is set.

        Parameters
        ----------
        shutdown : asyncio.Event
            Event to signal shutdown.
        """
        self.logger.info('Starting await_task loop')

        while not shutdown.is_set():
            try:
                # Wait for a task with timeout to check shutdown periodically
                try:
                    metadata = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Mark as busy
                self.is_busy = True

                # Run the simulation
                result = await self.run_simulation(metadata)

                # Store result as current task
                self.current_task = result

                # Mark as available
                self.is_busy = False

                self.logger.info(
                    f'Completed task {metadata.get("simulation_id")}',
                )

            except Exception as e:
                self._log_error('await_task', e)
                self.is_busy = False

        self.logger.info('Exiting await_task loop')


class SimulationPoolAgent(AcademyAgent):
    """Agent that manages a pool of simulation workers.

    This agent coordinates multiple SimulationAgent workers, distributing
    simulation tasks across them with load balancing and fault tolerance.

    Attributes
    ----------
    config : SimulationPoolConfig
        Configuration for the simulation pool.
    workers : list[Handle[SimulationAgent]]
        List of simulation worker agent handles.
    """

    def __init__(
        self,
        config: SimulationPoolConfig,
        workers: list[Handle[SimulationAgent]],
    ) -> None:
        """Initialize the simulation pool agent.

        Parameters
        ----------
        config : SimulationPoolConfig
            Configuration for the simulation pool.
        workers : list[Handle[SimulationAgent]]
            List of simulation worker agent handles.
        """
        super().__init__()
        self.config = config
        self.workers = workers
        self._pending_tasks: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._results: dict[str, dict[str, Any]] = {}
        self._task_retries: dict[str, int] = {}

    @action
    async def submit_simulation(
        self,
        metadata: dict[str, Any],
    ) -> str:
        """Submit a simulation to the pool.

        Parameters
        ----------
        metadata : dict[str, Any]
            Simulation metadata.

        Returns
        -------
        str
            Simulation ID for tracking.
        """
        sim_id = metadata.get('simulation_id', 'unknown')
        self._log_action('submit_simulation', sim_id=sim_id)

        # Initialize retry counter
        self._task_retries[sim_id] = 0

        # Add to pending queue
        await self._pending_tasks.put(metadata)

        self.logger.info(f'Submitted simulation {sim_id} to pool')
        return sim_id

    @action
    async def get_available_workers(self) -> list[int]:
        """Get indices of available workers.

        Returns
        -------
        list[int]
            List of worker indices that are available.
        """
        available = []

        for i, worker in enumerate(self.workers):
            try:
                is_available = await worker.is_available()
                if is_available:
                    available.append(i)
            except Exception as e:
                self._log_error('get_available_workers', e, worker_id=i)

        return available

    @action
    async def scale_pool(self, n_workers: int) -> None:
        """Scale the worker pool to the specified size.

        Note: This is a placeholder for Phase 2. Full implementation
        would require dynamic agent spawning/termination.

        Parameters
        ----------
        n_workers : int
            Target number of workers.
        """
        self._log_action('scale_pool', n_workers=n_workers)

        current_workers = len(self.workers)

        if n_workers > current_workers:
            self.logger.warning(
                f'Scaling up from {current_workers} to {n_workers} workers '
                f'not yet implemented. This requires dynamic agent spawning.',
            )
        elif n_workers < current_workers:
            self.logger.warning(
                f'Scaling down from {current_workers} to {n_workers} workers '
                f'not yet implemented. This requires graceful agent shutdown.',
            )
        else:
            self.logger.info(f'Pool already at target size: {n_workers}')

    @action
    async def get_result(self, sim_id: str) -> dict[str, Any] | None:
        """Get the result of a completed simulation.

        Parameters
        ----------
        sim_id : str
            Simulation ID.

        Returns
        -------
        dict[str, Any] | None
            Simulation result or None if not yet complete.
        """
        return self._results.get(sim_id)

    @action
    async def get_all_results(self) -> dict[str, dict[str, Any]]:
        """Get all completed simulation results.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping simulation IDs to results.
        """
        return self._results.copy()

    @action
    async def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        self._task_retries.clear()
        self.logger.info('Cleared all results')

    @loop
    async def load_balance(self, shutdown: asyncio.Event) -> None:
        """Distribute simulation tasks across available workers.

        This loop continuously monitors the pending task queue and
        assigns tasks to available workers with fault tolerance and
        automatic retry logic.

        Parameters
        ----------
        shutdown : asyncio.Event
            Event to signal shutdown.
        """
        self.logger.info('Starting load_balance loop')

        while not shutdown.is_set():
            try:
                # Check for pending tasks
                if self._pending_tasks.empty():
                    await asyncio.sleep(0.5)
                    continue

                # Find available workers
                available_workers = await self.get_available_workers()

                if not available_workers:
                    await asyncio.sleep(0.5)
                    continue

                # Get next task
                try:
                    metadata = await asyncio.wait_for(
                        self._pending_tasks.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                sim_id = metadata.get('simulation_id', 'unknown')

                # Select worker (simple round-robin for now)
                worker_idx = available_workers[0]
                worker = self.workers[worker_idx]

                self.logger.info(
                    f'Assigning simulation {sim_id} to worker {worker_idx}',
                )

                # Submit to worker
                try:
                    result = await worker.run_simulation(metadata)

                    # Check if simulation succeeded
                    if result.get('success', False):
                        # Store result
                        self._results[sim_id] = result
                        self.logger.info(
                            f'Simulation {sim_id} completed successfully',
                        )
                    else:
                        # Handle failure with retry logic
                        await self._handle_failed_simulation(metadata, result)

                except Exception as e:
                    self._log_error(
                        'load_balance',
                        e,
                        sim_id=sim_id,
                        worker_id=worker_idx,
                    )

                    # Handle failure with retry logic
                    await self._handle_failed_simulation(
                        metadata,
                        {'success': False, 'error': str(e)},
                    )

            except Exception as e:
                self._log_error('load_balance', e)
                await asyncio.sleep(1.0)

        self.logger.info('Exiting load_balance loop')

    async def _handle_failed_simulation(
        self,
        metadata: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Handle a failed simulation with retry logic.

        Parameters
        ----------
        metadata : dict[str, Any]
            Simulation metadata.
        result : dict[str, Any]
            Failed simulation result.
        """
        sim_id = metadata.get('simulation_id', 'unknown')
        retry_count = self._task_retries.get(sim_id, 0)

        if retry_count < self.config.max_retries:
            # Retry the simulation
            self._task_retries[sim_id] = retry_count + 1

            self.logger.warning(
                f'Simulation {sim_id} failed (attempt {retry_count + 1}/'
                f'{self.config.max_retries}). Retrying after '
                f'{self.config.retry_delay}s...',
            )

            # Wait before retrying
            await asyncio.sleep(self.config.retry_delay)

            # Re-queue the task
            await self._pending_tasks.put(metadata)

        else:
            # Max retries exceeded, store failed result
            self.logger.error(
                f'Simulation {sim_id} failed after {self.config.max_retries} '
                f'attempts. Giving up.',
            )

            self._results[sim_id] = result

