"""DDWE workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import result_processor
from proxystore.proxy import extract

from deepdrivewe import EnsembleCheckpointer
from deepdrivewe import WeightedEnsemble
from deepdrivewe.workflows.utils import ProxyManager
from deepdrivewe.workflows.utils import ResultLogger


class DDWEThinker(BaseThinker):
    """A thinker for the DDWE workflow."""

    def __init__(  # noqa: PLR0913
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
        use_stale_model: bool = False,
        streaming: bool = False,
        max_retries: int = 2,
        ps_name: str | None = None,
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
        streaming: bool
            Whether to stream the simulation results directly to the
            training task (default to False).
        max_retries: int
            Number of times to retry a task if it fails (default to 2).
        ps_name: str, optional
            The name of the proxy store to use, by default a No-op store.
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations
        self.use_stale_model = use_stale_model
        self.streaming = streaming
        self.max_retries = max_retries
        self.result_logger = ResultLogger(result_dir)

        # Store the simulation output (the input of both train/inference tasks)
        self.sim_output: list[Any] = []
        # Store the train output (the input of the inference task)
        self.train_output: Any = None

        # Setup manual proxies to control the evict policy
        self.proxy_manager = ProxyManager(ps_name)

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

        # If we are not streaming, then we need to submit a single train task
        # at the start of the workflow
        if self.streaming:
            self.logger.info('Start streaming train task')
            self.submit_task('train')

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
        # If we are streaming, then the simulation results only need
        # to be used to submit and inference task.
        output = result.value if self.streaming else extract(result.value)
        self.sim_output.append(output)

        # If we have all the simulation results, submit a train task
        if len(self.sim_output) == len(self.ensemble.next_sims):
            # Manually proxy the output objects to avoid auto-eviction
            # until the inference task is done (since both train/inference
            # tasks use the simulation output)
            # sim_proxy = self.proxy_manager.proxy(self.sim_output)

            # If we are streaming, then the simulation results are
            # directly routed to the training task via ProxyStream.
            # So, we don't need to submit an extra training task.
            if not self.streaming:
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

        # Store the training output
        self.train_output = result.value

        # Submit an inference task with the simulation/train task outputs
        if not self.streaming:
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
        # and clean up the proxy objects
        self.sim_output = []
        self.proxy_manager.evict()

        # Check if the workflow is finished (if so return before submitting)
        if self.ensemble.iteration >= self.num_iterations:
            self.logger.info('Workflow finished')
            self.done.set()
            return

        # Submit the next iteration of simulations
        self.logger.info('Submitting next iteration of simulations')
        for sim in self.ensemble.next_sims:
            self.submit_task('simulation', sim)
