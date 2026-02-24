"""Inference agent for weighted ensemble resampling and next-iteration dispatch."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from academy.agent import action
from academy.agent import loop
from academy.handle import Handle

from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TrainResult
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.binners.base import Binner
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.recyclers.base import Recycler
from deepdrivewe.resamplers.base import Resampler

if TYPE_CHECKING:
    from deepdrivewe.academy_agents.simulation import SimulationAgent


class InferenceAgentConfig:
    """Configuration for the InferenceAgent.

    Parameters
    ----------
    output_dir : Path
        Directory to store inference outputs and model state.
    pretrained_model_path : Path | None
        Path to a pretrained CVAE model checkpoint to load on startup.
        Having a pretrained model ensures the inference agent is ready
        immediately at iteration 1 without waiting for the training agent.
    cvae_config : ConvolutionalVAEConfig | None
        Configuration for the CVAE model used during inference (predict step).
        If None, default settings are used.
    """

    def __init__(
        self,
        output_dir: Path,
        pretrained_model_path: Path | None = None,
        cvae_config: object | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.pretrained_model_path = pretrained_model_path
        self.cvae_config = cvae_config


class InferenceAgent(AcademyAgent):
    """Agent that runs inference and drives the weighted ensemble iteration loop.

    This agent is the workflow coordinator in the decentralized Academy
    architecture. It:

    1. Collects SimResult objects from all N SimulationAgents
    2. Receives updated model weights from the TrainingAgent
    3. Runs CVAE inference (latent space projection) on collected data
    4. Applies WE resampling (binning, recycling, splitting/merging)
    5. Saves the ensemble checkpoint
    6. Dispatches the next iteration's SimMetadata to each SimulationAgent
    7. Signals shutdown when ``max_iterations`` is reached

    This mirrors the InferenceAgent from the minimal_pattern example
    (https://github.com/braceal/deepdrivewe-academy), extended with real
    CVAE inference and weighted ensemble resampling logic.

    The model stays warm in GPU memory across iterations because it is loaded
    in ``agent_on_startup()`` and kept as an instance attribute for the
    lifetime of the agent process.

    Attributes
    ----------
    num_simulations : int
        Number of SimulationAgents to collect results from per iteration.
    max_iterations : int
        Total number of WE iterations to run before shutting down.
    simulation_handles : list[Handle[SimulationAgent]]
        Handles to each SimulationAgent (used to dispatch next iteration).
    config : InferenceAgentConfig
        Configuration for the inference agent.
    binner : Binner
        WE binner for assigning simulations to bins.
    resampler : Resampler
        WE resampler for splitting/merging trajectories.
    recycler : Recycler
        WE recycler for handling terminal states.
    ensemble : WeightedEnsemble
        The current weighted ensemble state.
    checkpointer : EnsembleCheckpointer
        Checkpointer for saving ensemble state to disk after each iteration.
    """

    # Private state (not serialized, initialized in agent_on_startup)
    __logger: logging.Logger
    __batch: list[SimResult]
    __batch_ready: asyncio.Event
    __model_lock: asyncio.Lock

    def __init__(
        self,
        num_simulations: int,
        max_iterations: int,
        simulation_handles: list[Handle[SimulationAgent]],
        config: InferenceAgentConfig,
        binner: Binner,
        resampler: Resampler,
        recycler: Recycler,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
    ) -> None:
        """Initialize the inference agent.

        Parameters
        ----------
        num_simulations : int
            Number of simulation agents (batch size per iteration).
        max_iterations : int
            Total WE iterations to run.
        simulation_handles : list[Handle[SimulationAgent]]
            Handles for dispatching next-iteration work to each SimulationAgent.
        config : InferenceAgentConfig
            Configuration for the inference agent.
        binner : Binner
            WE binner.
        resampler : Resampler
            WE resampler.
        recycler : Recycler
            WE recycler.
        ensemble : WeightedEnsemble
            Initial weighted ensemble state (may be loaded from checkpoint).
        checkpointer : EnsembleCheckpointer
            Checkpointer to save ensemble state after each iteration.
        """
        super().__init__()
        self.num_simulations = num_simulations
        self.max_iterations = max_iterations
        self.simulation_handles = simulation_handles
        self.config = config
        self.binner = binner
        self.resampler = resampler
        self.recycler = recycler
        self.ensemble = ensemble
        self.checkpointer = checkpointer

    async def agent_on_startup(self) -> None:
        """Initialize state and load the pretrained CVAE model onto GPU.

        All stateful initialization happens here so it runs on the correct
        worker process (i.e., the GPU node where this agent is placed by
        the ParslPoolExecutor). This ensures the model is warm in GPU memory
        before the first iteration starts.
        """
        self.__logger = logging.getLogger(self.__class__.__name__)  # type: ignore[misc]
        self.__batch = []
        self.__batch_ready = asyncio.Event()
        self.__model_lock = asyncio.Lock()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load the CVAE model for inference (lazy import for HPC compatibility)
        try:
            from deepdrivewe.ai.cvae import ConvolutionalVAE
            from deepdrivewe.ai.cvae import ConvolutionalVAEConfig

            cvae_config = (
                self.config.cvae_config
                if self.config.cvae_config is not None
                else ConvolutionalVAEConfig()
            )

            self.__model = ConvolutionalVAE(  # type: ignore[misc]
                config=cvae_config,
                checkpoint_path=self.config.pretrained_model_path,
            )

            self.__logger.info(
                'CVAE inference model loaded'
                + (
                    f' from {self.config.pretrained_model_path}'
                    if self.config.pretrained_model_path
                    else ' (initialized from scratch)'
                ),
            )
        except ImportError as e:
            self.__logger.warning(
                f'Could not import CVAE model dependencies: {e}. '
                'Running in mock mode (no latent projection during inference).',
            )
            self.__model = None  # type: ignore[misc]

        self.__logger.info(
            f'InferenceAgent started. Will run {self.max_iterations} iterations '
            f'with {self.num_simulations} simulation(s) per iteration.',
        )

    @action
    async def receive_simulation_data(self, result: SimResult) -> None:
        """Receive one SimResult and buffer it for the current batch.

        Called by each SimulationAgent after completing its run. When
        ``num_simulations`` results have been received, the ``__batch_ready``
        event is set to trigger the inference loop.

        Parameters
        ----------
        result : SimResult
            Completed simulation result (trajectory data + metadata).
        """
        self.__logger.info(
            f'Received result for sim {result.metadata.simulation_id} '
            f'iteration {result.metadata.iteration_id}. '
            f'Batch: {len(self.__batch) + 1}/{self.num_simulations}',
        )
        self.__batch.append(result)

        # Signal the infer loop when all results are collected
        if len(self.__batch) >= self.num_simulations:
            self.__batch_ready.set()

    @action
    async def receive_model_weights(self, train_result: TrainResult) -> None:
        """Receive updated model weights from the TrainingAgent.

        Updates the CVAE model weights used for latent space inference.
        An async lock guards model updates to avoid races with the infer loop.

        Parameters
        ----------
        train_result : TrainResult
            Result from a training step, containing the checkpoint path.
        """
        self.__logger.info(
            f'Received updated model weights: {train_result.checkpoint_path}',
        )
        async with self.__model_lock:
            if self.__model is not None:  # type: ignore[misc]
                try:
                    await asyncio.to_thread(
                        self.__model.update_model,  # type: ignore[misc]
                        train_result.checkpoint_path,
                    )
                    self.__logger.info('Model weights updated successfully')
                except Exception as e:
                    self._log_error('receive_model_weights', e)

    @loop
    async def infer(self, shutdown: asyncio.Event) -> None:
        """Wait for a full batch then run inference and advance the WE iteration.

        This is the main driver loop of the entire workflow. For each iteration:
        1. Waits until all N simulation results are collected
        2. Runs CVAE latent projection on contact maps (under model lock)
        3. Applies WE resampling (bin → recycle → split/merge)
        4. Saves the ensemble checkpoint
        5. Dispatches the next iteration's SimMetadata to each SimulationAgent
        6. Shuts down when max_iterations is reached

        Parameters
        ----------
        shutdown : asyncio.Event
            Event set by the Academy runtime when the agent should stop.
        """
        self.__logger.info('Inference loop started')

        while not shutdown.is_set():
            # Wait until all simulation results are collected
            try:
                await asyncio.wait_for(
                    self.__batch_ready.wait(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                # Check shutdown periodically
                continue

            self.__batch_ready.clear()

            # Grab the current batch and reset for next iteration
            batch = self.__batch
            self.__batch = []

            current_iteration = self.ensemble.iteration
            self.__logger.info(
                f'Running inference on {len(batch)} results for '
                f'iteration {current_iteration}',
            )

            try:
                # Step 1: Run CVAE latent projection (updates auxdata in-place)
                async with self.__model_lock:
                    if self.__model is not None:  # type: ignore[misc]
                        await asyncio.to_thread(
                            self._project_to_latent,
                            batch,
                        )

                # Step 2: Extract SimMetadata from results (with pcoords populated)
                cur_sims = [result.metadata for result in batch]

                # Step 3: Apply WE resampling pipeline
                cur_sims_out, next_sims, iteration_metadata = (
                    await asyncio.to_thread(
                        self.resampler.run,
                        cur_sims,
                        self.binner,
                        self.recycler,
                    )
                )

                # Step 4: Advance ensemble state
                self.ensemble.advance_iteration(
                    cur_sims=cur_sims_out,
                    next_sims=next_sims,
                    metadata=iteration_metadata,
                )

                # Step 5: Save checkpoint
                await asyncio.to_thread(self.checkpointer.save, self.ensemble)

                self.__logger.info(
                    f'Iteration {current_iteration} complete. '
                    f'Next iteration: {len(next_sims)} simulations.',
                )

            except Exception as e:
                self._log_error('infer', e)
                # Do not shut down on error — log and continue waiting
                # for the next batch (simulations may retry)
                continue

            # Check if we have reached the maximum number of iterations
            if current_iteration >= self.max_iterations:
                self.__logger.info(
                    f'Reached max iterations ({self.max_iterations}), '
                    'shutting down.',
                )
                shutdown.set()
                return

            # Step 6: Dispatch the next iteration of simulations
            # next_sims may have more or fewer entries than simulation_handles
            # (due to splitting/merging). We cycle through handles if needed.
            self.__logger.info(
                f'Kicking off iteration {current_iteration + 1} '
                f'with {len(next_sims)} simulations.',
            )

            dispatch_tasks = []
            for idx, sim_meta in enumerate(next_sims):
                # Round-robin across available simulation handles
                handle = self.simulation_handles[idx % len(self.simulation_handles)]
                dispatch_tasks.append(handle.simulate(sim_meta))

            # Dispatch all simulations concurrently
            await asyncio.gather(*dispatch_tasks)

        self.__logger.info('Inference loop exited')

    def _project_to_latent(self, batch: list[SimResult]) -> None:
        """Run CVAE latent projection and store embeddings in SimResult auxdata.

        This is a synchronous method executed in a thread (via asyncio.to_thread)
        so that GPU computation does not block the event loop.

        Parameters
        ----------
        batch : list[SimResult]
            Batch of simulation results. Contact maps are read from
            ``result.data['contact_maps']`` and latent embeddings are
            stored back into ``result.metadata.auxdata['latent_embeddings']``.
        """
        import numpy as np

        for result in batch:
            contact_maps = result.data.get('contact_maps')

            if contact_maps is None or len(contact_maps) == 0:
                self.__logger.debug(
                    f'No contact maps for sim {result.metadata.simulation_id}',
                )
                continue

            # Ensure correct dtype / shape for CVAE
            x = np.array(contact_maps)

            # Run prediction (n_frames, latent_dim)
            try:
                embeddings = self.__model.predict(x)  # type: ignore[misc]
                result.metadata.auxdata['latent_embeddings'] = (
                    embeddings.tolist()
                )
            except Exception as e:
                self.__logger.warning(
                    f'CVAE prediction failed for sim '
                    f'{result.metadata.simulation_id}: {e}',
                )
