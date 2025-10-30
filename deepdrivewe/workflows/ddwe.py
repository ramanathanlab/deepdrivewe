"""DDWE workflow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import result_processor
from proxystore.proxy import extract
from proxystore.store.utils import get_key

from deepdrivewe import EnsembleCheckpointer
from deepdrivewe import WeightedEnsemble
from deepdrivewe.workflows.stream import ProxyStreamConfig
from deepdrivewe.workflows.stream import SIMULATION_TOPIC
from deepdrivewe.workflows.stream import TRAIN_TOPIC
from deepdrivewe.workflows.utils import ResultLogger


class DDWEThinker(BaseThinker):
    """A thinker for the DDWE workflow."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
        use_stale_model: bool = False,
        max_retries: int = 2,
    ) -> None:
        """Initialize the DDWE workflow thinker.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server.
        result_dir: Path
            Directory in which to store outputs.
        ensemble: WeightedEnsemble
            The weighted ensemble to use for the workflow.
        checkpointer: EnsembleCheckpointer
            Checkpointer for the weighted ensemble.
        num_iterations: int
            Number of iterations to run the workflow.
        use_stale_model: bool
            Whether to use the stale model for inference (default to False).
            This will be faster but may not be as accurate. It uses the
            model from the previous iteration for inference in the current
            iteration, which may not be updated with new states.
        max_retries: int
            Number of times to retry a task if it fails (default to 2).
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations
        self.use_stale_model = use_stale_model
        self.max_retries = max_retries
        self.result_logger = ResultLogger(result_dir)

        # Store the simulation output (the input of both train/inference tasks)
        self.sim_output: list[Any] = []
        # Store the train output (the input of the inference task)
        self.train_output: Any = None

    def submit_task(self, topic: str, *inputs: Any) -> None:
        """Submit a task to the task server.

        Parameters
        ----------
        topic: str
            The topic of the task.
        inputs: Any
            The input args to the task.
        """
        # Submit the task to the task server
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            max_retries=self.max_retries,
        )

    @agent(startup=True)
    def start_workflow(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        # Submit the next iteration of simulations
        for sim in self.ensemble.next_sims:
            self.submit_task('simulation', sim)

    @result_processor(topic='simulation')
    def process_simulation_result(self, result: Result) -> None:
        """Process a simulation result."""
        # Log simulation job results
        self.result_logger.log(result, topic='simulation')

        # Check if the task failed
        if not result.success:
            self.logger.error(
                f'Simulation failed after {result.retries}'
                f'/{result.max_retries} attempts, quitting workflow.',
                f' result={result}',
            )
            self.done.set()
            return

        # Collect simulation results for the current iteration
        # Note: We need to extract the proxied objects before storing them
        # to avoid auto-eviction after single use. The return results
        # are re-proxied before submitting the train/inference tasks.
        self.sim_output.append(extract(result.value))

        # If we have all the simulation results, submit a train task
        if len(self.sim_output) == len(self.ensemble.next_sims):
            # Submit the train task
            self.submit_task('train', self.sim_output)
            self.logger.info('Submitting training task')

            # If it's okay to use the stale model, submit the inference task
            # using the previous iteration's model
            if self.use_stale_model and self.train_output is not None:
                self.submit_task(
                    'inference',
                    self.sim_output,
                    self.train_output,
                )

    @result_processor(topic='train')
    def process_train_result(self, result: Result) -> None:
        """Process a training result."""
        # Log training job results
        self.result_logger.log(result, topic='train')

        # Check if the task failed
        if not result.success:
            self.logger.warning('Training failed, quitting workflow.')
            self.done.set()
            return

        # See if this is the first training task return value
        first_train = self.train_output is None

        # Store the training output
        self.train_output = result.value

        # Submit an inference task with the simulation/train task outputs
        if first_train or not self.use_stale_model:
            self.submit_task('inference', self.sim_output, self.train_output)
            self.logger.info('submitted inference task')

    @result_processor(topic='inference')
    def process_inference_result(self, result: Result) -> None:
        """Process an inference result."""
        # Log inference job results
        self.result_logger.log(result, topic='inference')

        # Check if the task failed
        if not result.success:
            self.logger.warning('Inference failed, quitting workflow.')
            self.done.set()
            return

        # Unpack the output
        cur_sims, next_sims, metadata = result.value

        # Update the weighted ensemble with the next iteration
        self.ensemble.advance_iteration(
            cur_sims=cur_sims,
            next_sims=next_sims,
            metadata=metadata,
        )

        # Save an ensemble checkpoint
        self.checkpointer.save(self.ensemble)

        # Log the current iteration
        self.logger.info(f'Current iteration: {self.ensemble.iteration}')

        # Reset the simulation output for the next iteration
        self.sim_output = []

        # Check if the workflow is finished (if so return before submitting)
        if self.ensemble.iteration >= self.num_iterations:
            self.logger.info('Workflow finished')
            self.done.set()
            return

        # Submit the next iteration of simulations
        self.logger.info('Submitting next iteration of simulations')
        for sim in self.ensemble.next_sims:
            self.submit_task('simulation', sim)


class DDWEStreamThinker(BaseThinker):
    """A thinker for the DDWE workflow."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
        stream_config: ProxyStreamConfig,
        use_stale_model: bool = False,
        max_retries: int = 2,
    ) -> None:
        """Initialize the DDWE workflow thinker.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server.
        result_dir: Path
            Directory in which to store outputs.
        ensemble: WeightedEnsemble
            The weighted ensemble to use for the workflow.
        checkpointer: EnsembleCheckpointer
            Checkpointer for the weighted ensemble.
        num_iterations: int
            Number of iterations to run the workflow.
        stream_config: ProxyStreamConfig
            Configuration for the data stream.
        use_stale_model: bool
            Whether to use the stale model for inference (default to False).
            This will be faster but may not be as accurate. It uses the
            model from the previous iteration for inference in the current
            iteration, which may not be updated with new states.
        max_retries: int
            Number of times to retry a task if it fails (default to 2).
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations
        self.stream_config = stream_config
        self.use_stale_model = use_stale_model
        self.max_retries = max_retries
        self.result_logger = ResultLogger(result_dir)

        # Store the simulation output (the input of both train/inference tasks)
        self.sim_output: list[Any] = []

        # TODO: These two attributes need to be checkpointed and restored
        # Store the train output (the input of the inference task)
        self.train_output: Any = None
        # Keep a counter for the current training iteration
        self.train_iteration = ensemble.iteration - 1

        # Create a consumer for streaming the training return objects
        # to the thinker.
        self.stream_config = stream_config
        self.stream_consumer = stream_config.get_consumer(topic=TRAIN_TOPIC)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        """Submit a task to the task server.

        Parameters
        ----------
        topic: str
            The topic of the task.
        inputs: Any
            The input args to the task.
        """
        # Submit the task to the task server
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            max_retries=self.max_retries,
        )

    @agent(startup=True)
    def start_workflow(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        # Submit the next iteration of simulations
        for sim in self.ensemble.next_sims:
            self.submit_task('simulation', sim)

        # We need to submit a single train task at the start of the workflow
        # to kick off the simulation stream consumer. We send an empty list
        # of simulation outputs to be compatible with the train task signature.
        self.logger.info('Start streaming train task')
        self.submit_task('train', [])

    @result_processor(topic='simulation')
    def process_simulation_result(self, result: Result) -> None:
        """Process a simulation result."""
        # Log simulation job results
        self.result_logger.log(result, topic='simulation')

        # Check if the task failed
        if not result.success:
            self.logger.error(
                f'Simulation failed after {result.retries}'
                f'/{result.max_retries} attempts, quitting workflow.',
                f' result={result}',
            )
            self.stop_workflow()
            return

        # Collect simulation results for the current iteration
        self.sim_output.append(result.value)

        # If we have all the simulation results, submit the inference task
        # using the previous iteration's model
        if len(self.sim_output) == len(self.ensemble.next_sims):
            # We need to wait for the first streaming train task to finish
            if self.use_stale_model and self.train_output is None:
                # Wait for the streaming train task to finish
                self.logger.info(
                    'Waiting for first streaming train task to finish',
                )
                while self.train_output is None:
                    time.sleep(10)

            # We need to wait for the next streaming train task to finish
            # to get a fresh model
            elif not self.use_stale_model:
                self.logger.info(
                    'Waiting for next streaming train task to finish',
                )
                while self.train_iteration < self.ensemble.iteration:
                    time.sleep(10)
                # This should hold (see train_stream_processor)
                assert self.train_output is not None

            # Submit the inference task using either a stale or fresh model
            self.submit_task('inference', self.sim_output, self.train_output)

    @agent()
    def train_stream_processor(self) -> None:
        """Process the streaming train task."""
        # This for loop will run until the producer closes the topic
        # (see stop_workflow)
        for result in self.stream_consumer:
            # Log a message for each train result
            self.logger.info('Received streaming train result')

            # Clean up the previous training output from the store
            if self.train_output is not None:
                self.logger.info('Evicting previous training output')
                # TODO: Does this raise an error since evict=True in run_train?
                # Get the proxy key for the current training output
                key = get_key(self.train_output)
                # Evict the key from the store to clean up memory
                self.stream_config.get_store().evict(key)
                self.logger.info(
                    f'Evicted previous training output with key: {key}',
                )

            # Store the training output
            self.train_output = result

            # Increment the training iteration
            self.train_iteration += 1

    def stop_workflow(self) -> None:
        """Stop the workflow."""
        # Set the done flag to signal the agents to stop
        self.done.set()

        # Close the stream consumer (we use the producer to close the topic)
        # NOTE: Closing the train topic, will close the stream_consumer in the
        # thinker which will stop the train_stream_processor agent, and closing
        # the simulation topic will close the training function consumer
        # waiting for new simulation results.
        for topic in [TRAIN_TOPIC, SIMULATION_TOPIC]:
            self.stream_config.get_producer(topic=topic).close_topics(topic)

        # Log a message that the workflow is stopping
        self.logger.info('Stopping the workflow')

    @result_processor(topic='inference')
    def process_inference_result(self, result: Result) -> None:
        """Process an inference result."""
        # Log inference job results
        self.result_logger.log(result, topic='inference')

        # Check if the task failed
        if not result.success:
            self.logger.error('Inference failed, quitting workflow.')
            self.stop_workflow()
            return

        # Unpack the output
        cur_sims, next_sims, metadata = result.value

        # Update the weighted ensemble with the next iteration
        self.ensemble.advance_iteration(
            cur_sims=cur_sims,
            next_sims=next_sims,
            metadata=metadata,
        )

        # Save an ensemble checkpoint
        self.checkpointer.save(self.ensemble)

        # Log the current iteration
        self.logger.info(f'Current iteration: {self.ensemble.iteration}')

        # Reset the simulation output for the next iteration
        self.sim_output = []

        # Check if the workflow is finished (if so return before submitting)
        if self.ensemble.iteration >= self.num_iterations:
            self.logger.info('Workflow finished')
            self.done.set()
            return

        # Submit the next iteration of simulations
        self.logger.info('Submitting next iteration of simulations')
        for sim in self.ensemble.next_sims:
            self.submit_task('simulation', sim)
