"""Simulation agents for running MD simulations."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from academy.agent import action
from academy.agent import loop
from academy.handle import Handle

from deepdrivewe import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.academy_agents.config import SimulationPoolConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation

if TYPE_CHECKING:
    from deepdrivewe.academy_agents.training import TrainingAgent
    from deepdrivewe.academy_agents.inference import InferenceAgent


class SimulationAgent(AcademyAgent):
    """Agent that executes individual MD simulations.

    In the decentralized Academy architecture, each SimulationAgent is its
    own actor. It receives ``SimMetadata`` via the ``simulate`` action,
    runs the OpenMM simulation, and streams the ``SimResult`` directly to
    both the TrainingAgent and the InferenceAgent — no central orchestrator
    is involved.

    This mirrors the SimulationAgent from the minimal_pattern example
    (https://github.com/braceal/deepdrivewe-academy), extended with real
    OpenMM simulation logic.

    Attributes
    ----------
    config : SimulationPoolConfig
        Configuration for simulations (output dir, OpenMM settings, etc.).
    train_handle : Handle[TrainingAgent] | None
        Handle to the TrainingAgent. When set, the SimResult is streamed
        directly after each simulation completes.
    inference_handle : Handle[InferenceAgent] | None
        Handle to the InferenceAgent. When set, the SimResult is sent
        directly after each simulation completes.
    """

    # Private logger (not serialized)
    __logger: logging.Logger

    def __init__(
        self,
        config: SimulationPoolConfig,
        train_handle: Handle[TrainingAgent] | None = None,
        inference_handle: Handle[InferenceAgent] | None = None,
    ) -> None:
        """Initialize the simulation agent.

        Parameters
        ----------
        config : SimulationPoolConfig
            Configuration for simulations.
        train_handle : Handle[TrainingAgent] | None
            Handle to the TrainingAgent for streaming simulation results.
            If None, results are not forwarded (pool-based mode).
        inference_handle : Handle[InferenceAgent] | None
            Handle to the InferenceAgent for streaming simulation results.
            If None, results are not forwarded (pool-based mode).
        """
        super().__init__()
        self.config = config
        self.train_handle = train_handle
        self.inference_handle = inference_handle

        # Legacy pool-based state (kept for backwards compatibility)
        self.current_task: dict[str, Any] | None = None
        self.is_busy = False
        self._task_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

    async def agent_on_startup(self) -> None:
        """Initialize the agent logger."""
        self.__logger = logging.getLogger(self.__class__.__name__)  # type: ignore[misc]
        self.__logger.info('SimulationAgent started')

    @action
    async def simulate(self, sim_metadata: SimMetadata) -> None:
        """Run a simulation and send the result to the TrainingAgent and InferenceAgent.

        This is the primary entry point in the decentralized Academy pattern.
        It is called by the InferenceAgent at the start of each iteration
        (or by ``main()`` to kick off iteration 1). After the simulation
        completes, the ``SimResult`` is forwarded simultaneously to both
        the TrainingAgent (for online model training) and the InferenceAgent
        (for batch collection and WE resampling).

        This matches the ``simulate`` action in the minimal_pattern example.

        Parameters
        ----------
        sim_metadata : SimMetadata
            Metadata describing the simulation to run (parent restart file,
            weights, iteration ID, etc.).
        """
        self.__logger.info(  # type: ignore[misc]
            f'Running simulation {sim_metadata.simulation_id} '
            f'iteration {sim_metadata.iteration_id}',
        )

        # Execute the simulation and get the raw result dict
        result_dict = await self.run_simulation(sim_metadata.model_dump())

        # Build the SimResult dataclass from the result
        updated_metadata = SimMetadata(**result_dict['metadata'])

        # Collect trajectory-derived data arrays
        contact_maps = result_dict.get('contact_maps', np.array([]))
        rmsd = result_dict.get('rmsd', np.array([]))

        sim_result = SimResult(
            data={
                'contact_maps': np.array(contact_maps),
                'rmsd': np.array(rmsd),
            },
            metadata=updated_metadata,
        )

        self.__logger.info(  # type: ignore[misc]
            f'Simulation {sim_metadata.simulation_id} complete, '
            f'forwarding result to training and inference agents.',
        )

        # Stream directly to TrainingAgent and InferenceAgent (decentralized pattern)
        forward_tasks = []
        if self.train_handle is not None:
            forward_tasks.append(
                self.train_handle.receive_simulation_data(sim_result),
            )
        if self.inference_handle is not None:
            forward_tasks.append(
                self.inference_handle.receive_simulation_data(sim_result),
            )

        if forward_tasks:
            await asyncio.gather(*forward_tasks)

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

            # Create RMSD reporter for progress coordinate calculation if reference file is provided
            reporters = []
            if self.config.reference_file is not None:
                from deepdrivewe.simulation.openmm import ContactMapRMSDReporter

                reporter = ContactMapRMSDReporter(
                    report_interval=self.config.simulation_config.report_steps,
                    reference_file=self.config.reference_file,
                    cutoff_angstrom=self.config.cutoff_angstrom,
                    mda_selection=self.config.mda_selection,
                    openmm_selection=self.config.openmm_selection,
                )
                reporters.append(reporter)

            # Run the simulation (blocking operation)
            # We run this in a thread pool to avoid blocking the event loop
            await asyncio.to_thread(simulation.run, reporters=reporters)

            # Extract progress coordinate (RMSD values) and contact maps
            # from the reporter if one was configured.
            if reporters:
                pcoord = reporters[0].get_rmsds()
                contact_maps = reporters[0].get_contact_maps()
            else:
                # No progress coordinate / contact maps computed
                pcoord = []
                contact_maps = []

            # Get trajectory file paths
            trajectory_data = {
                'restart_file': str(simulation.restart_file),
                'trajectory_file': str(simulation.trajectory_file),
                'log_file': str(simulation.log_file),
            }

            # Update metadata with progress coordinate and contact map auxdata
            sim_metadata.restart_file = simulation.restart_file
            sim_metadata.pcoord = (
                pcoord.tolist() if hasattr(pcoord, 'tolist') else list(pcoord)
            )
            sim_metadata.mark_simulation_end()

            self.logger.info(
                f'Completed simulation {sim_metadata.simulation_id} '
                f'in {sim_metadata.walltime:.2f}s',
            )

            return {
                'metadata': sim_metadata.model_dump(),
                'trajectory': trajectory_data,
                # Data arrays used by the simulate() action to build SimResult
                'contact_maps': (
                    contact_maps.tolist()
                    if hasattr(contact_maps, 'tolist')
                    else list(contact_maps)
                ),
                'rmsd': (
                    pcoord.tolist()
                    if hasattr(pcoord, 'tolist')
                    else list(pcoord)
                ),
                'success': True,
            }

        except Exception as e:
            self._log_error('run_simulation', e, sim_id=metadata.get('simulation_id'))
            sim_metadata.mark_simulation_end()

            return {
                'metadata': sim_metadata.model_dump(),
                'trajectory': {},
                'contact_maps': [],
                'rmsd': [],
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

