"""Training agent for online CVAE model training on simulation data."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from academy.agent import action
from academy.agent import loop
from academy.handle import Handle

from deepdrivewe.api import SimResult
from deepdrivewe.api import TrainResult
from deepdrivewe.academy_agents.base import AcademyAgent

if TYPE_CHECKING:
    from deepdrivewe.academy_agents.inference import InferenceAgent


class TrainingAgentConfig:
    """Configuration for the TrainingAgent.

    Parameters
    ----------
    output_dir : Path
        Directory to store model checkpoints.
    pretrained_model_path : Path | None
        Path to a pretrained CVAE model checkpoint to load on startup.
        If None, the model will be initialized from scratch.
    train_frequency : int
        Number of SimResults to accumulate before triggering a training step.
    cvae_config : ConvolutionalVAEConfig | None
        Configuration for the CVAE model. If None, default settings are used.
    """

    def __init__(
        self,
        output_dir: Path,
        pretrained_model_path: Path | None = None,
        train_frequency: int = 1,
        cvae_config: object | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.pretrained_model_path = pretrained_model_path
        self.train_frequency = train_frequency
        self.cvae_config = cvae_config


class TrainingAgent(AcademyAgent):
    """Agent that trains the CVAE model on incoming simulation data.

    This agent runs on a GPU node and keeps the model warm in memory
    across iterations. It receives SimResult objects from SimulationAgents
    via its mailbox, accumulates them in an internal queue, and trains the
    CVAE when enough data has been collected.

    After each training step, it sends the path to the updated model
    checkpoint to the InferenceAgent.

    This agent mirrors the TrainingAgent pattern from the minimal_pattern
    example (https://github.com/braceal/deepdrivewe-academy), extended
    with real CVAE training logic.

    Attributes
    ----------
    config : TrainingAgentConfig
        Configuration for the training agent.
    inference_handle : Handle[InferenceAgent]
        Handle to the inference agent to send model weights to.
    """

    # Class-level type annotation for the private logger (not serialized)
    __logger: logging.Logger

    # Internal queue for receiving SimResult objects from simulation agents
    __queue: asyncio.Queue[SimResult]

    def __init__(
        self,
        inference_handle: Handle[InferenceAgent],
        config: TrainingAgentConfig,
    ) -> None:
        """Initialize the training agent.

        Parameters
        ----------
        inference_handle : Handle[InferenceAgent]
            Handle to the inference agent to send updated model weights.
        config : TrainingAgentConfig
            Configuration for the training agent.
        """
        super().__init__()
        self.inference_handle = inference_handle
        self.config = config

    async def agent_on_startup(self) -> None:
        """Initialize state and load the CVAE model onto GPU.

        This is called by the Academy runtime when the agent starts. All
        stateful initialization (model loading, queue creation) happens
        here rather than in __init__ to ensure it runs on the correct
        worker process (i.e., the GPU node where this agent is placed).
        """
        self.__logger = logging.getLogger(self.__class__.__name__)  # type: ignore[misc]
        self.__queue = asyncio.Queue()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load the CVAE model (lazy import to avoid requiring torch at
        # import time on the client / head node)
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
                'CVAE model loaded successfully'
                + (
                    f' from {self.config.pretrained_model_path}'
                    if self.config.pretrained_model_path
                    else ' (initialized from scratch)'
                ),
            )
        except ImportError as e:
            # mdlearn / torch not available — run in CPU-only / mock mode
            self.__logger.warning(
                f'Could not import CVAE model dependencies: {e}. '
                'Running in mock mode (no actual training will occur).',
            )
            self.__model = None  # type: ignore[misc]

        self.__logger.info('TrainingAgent started')

    @action
    async def receive_simulation_data(self, result: SimResult) -> None:
        """Receive a simulation result and queue it for training.

        This action is called by each SimulationAgent after completing
        a simulation run. The result is placed onto an internal async
        queue which is drained by the ``train`` loop.

        Parameters
        ----------
        result : SimResult
            The completed simulation result, including trajectory data
            (contact maps, RMSD) and metadata.
        """
        self.__logger.info(
            f'Received simulation data for sim '
            f'{result.metadata.simulation_id} '
            f'iteration {result.metadata.iteration_id}',
        )
        await self.__queue.put(result)

    @loop
    async def train(self, shutdown: asyncio.Event) -> None:
        """Drain the simulation queue and train the CVAE model.

        This loop runs continuously in the background. It accumulates
        ``config.train_frequency`` SimResult objects, then trains the CVAE
        on the collected contact maps. After training, it sends the path of
        the new model checkpoint to the InferenceAgent.

        The loop exits gracefully when the ``shutdown`` event is set.

        Parameters
        ----------
        shutdown : asyncio.Event
            Event set by the Academy runtime when the agent should stop.
        """
        self.__logger.info('Training loop started')

        while not shutdown.is_set():
            # Accumulate train_frequency results before training
            batch: list[SimResult] = []

            for _ in range(self.config.train_frequency):
                try:
                    result = await asyncio.wait_for(
                        self.__queue.get(),
                        timeout=1.0,
                    )
                    batch.append(result)
                    self.__queue.task_done()
                except asyncio.TimeoutError:
                    # Check shutdown and retry
                    if shutdown.is_set():
                        break
                    continue

            if not batch:
                continue

            self.__logger.info(
                f'Training on batch of {len(batch)} simulation results',
            )

            try:
                checkpoint_path = await asyncio.to_thread(
                    self._train_on_batch,
                    batch,
                )

                self.__logger.info(
                    f'Training complete. Checkpoint: {checkpoint_path}',
                )

                # Send updated model weights to the inference agent
                train_result = TrainResult(
                    config_path=self.config.output_dir / 'cvae_config.yaml',
                    checkpoint_path=checkpoint_path,
                )
                await self.inference_handle.receive_model_weights(train_result)

            except Exception as e:
                self._log_error('train', e)

        self.__logger.info('Training loop exited')

    def _train_on_batch(self, batch: list[SimResult]) -> Path:
        """Train the CVAE model on a batch of simulation results.

        This is a synchronous method run in a thread via asyncio.to_thread
        so that it does not block the event loop during GPU computation.

        Parameters
        ----------
        batch : list[SimResult]
            Batch of simulation results containing contact maps.

        Returns
        -------
        Path
            Path to the newly saved model checkpoint.
        """
        import numpy as np

        # Extract contact maps from the simulation results
        # Shape: list of (n_frames, n_atoms, n_atoms) arrays
        all_contact_maps = []
        all_rmsds = []

        for result in batch:
            contact_maps = result.data.get('contact_maps')
            rmsd = result.data.get('rmsd')

            if contact_maps is not None and len(contact_maps) > 0:
                all_contact_maps.append(contact_maps)

            if rmsd is not None and len(rmsd) > 0:
                all_rmsds.append(rmsd)

        if not all_contact_maps or self.__model is None:  # type: ignore[misc]
            # No contact map data or no model — save a placeholder checkpoint
            self.__logger.warning(
                'No contact map data available for training or model is None. '
                'Saving placeholder checkpoint.',
            )
            placeholder = self.config.output_dir / 'model_placeholder.pt'
            placeholder.touch()
            return placeholder

        # Stack all contact maps: (total_frames, n_atoms, n_atoms)
        x = np.concatenate(all_contact_maps, axis=0)
        scalars = {}

        if all_rmsds:
            scalars['rmsd'] = np.concatenate(all_rmsds, axis=0)

        # Determine model output directory for this training step
        iteration = batch[0].metadata.iteration_id
        model_dir = self.config.output_dir / f'cvae_iter_{iteration:06d}'
        model_dir.mkdir(parents=True, exist_ok=True)

        # Fit the CVAE and get the latest checkpoint path
        checkpoint_path = self.__model.fit(  # type: ignore[misc]
            x=x,
            model_dir=model_dir,
            scalars=scalars if scalars else None,
        )

        return checkpoint_path
