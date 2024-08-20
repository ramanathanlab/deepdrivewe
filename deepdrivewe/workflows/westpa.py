"""WESTPA workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import result_processor

from deepdrivewe import EnsembleCheckpointer
from deepdrivewe import WeightedEnsemble
from deepdrivewe.workflows.utils import ResultLogger


class WESTPAThinker(BaseThinker):
    """A synchronous DDWE thinker."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
        simulation_retry_limit: int = 2,
    ) -> None:
        """Initialize the synchronous DDWE thinker.

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
        simulation_retry_limit: int
            Number of times to retry a simulation task if it fails.
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations
        self.simulation_retry_limit = simulation_retry_limit
        self.result_logger = ResultLogger(result_dir)

        # Store the inference input (the output of the simulations)
        self.inference_input: list[Any] = []

    def submit_task(
        self,
        topic: str,
        *inputs: Any,
        keep_inputs: bool = False,
        retry_count: int | None = None,
    ) -> None:
        """Submit a task to the task server.

        Parameters
        ----------
        topic: str
            The topic of the task.
        inputs: Any
            The input args to the task.
        keep_inputs: bool
            Whether to keep the inputs in the task server.
        retry_count: int | None
            The number of times the task has been retried
            (default to 0 when None is specified).
        """
        # Initialize the retry count
        retry_count = 0 if retry_count is None else retry_count

        # Submit the task to the task server
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            keep_inputs=keep_inputs,
            task_info={'retry_count': retry_count},
        )

    @agent(startup=True)
    def start_simulations(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        # Submit the next iteration of simulations
        for sim in self.ensemble.current_sims:
            self.submit_task('simulation', sim, keep_inputs=True)

    @result_processor(topic='simulation')
    def process_simulation_result(self, result: Result) -> None:
        """Process a simulation result."""
        # Log simulation job results
        self.result_logger.log(result, topic='simulation')

        # Resubmit the simulation task if it failed. If
        # the task has failed too many times, then quit the workflow.
        if not result.success:
            retry_count: int = result.task_info['retry_count']
            if retry_count >= self.simulation_retry_limit:
                self.logger.error(
                    f'Simulation {result.task_id} '
                    f'failed {retry_count} times, quitting workflow.',
                )
                self.done.set()
                return

            self.submit_task(
                'simulation',
                *result.args,
                keep_inputs=True,
                retry_count=retry_count + 1,
            )
            self.logger.warning(
                f'Simulation {result.task_id} failed, resubmitted task',
            )
            return

        # Collect simulation results
        self.inference_input.append(result.value)

        if len(self.inference_input) == len(self.ensemble.current_sims):
            self.submit_task('inference', self.inference_input)
            self.inference_input = []  # Clear batched data
            self.logger.info('submitted inference task')

    @result_processor(topic='inference')
    def process_inference_result(self, result: Result) -> None:
        """Process an inference result."""
        # Log inference job results
        self.result_logger.log(result, topic='inference')
        if not result.success:
            self.logger.warning('Bad inference result, quitting workflow.')
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

        # Check if the workflow is finished (if so return before submitting)
        if self.ensemble.iteration >= self.num_iterations:
            self.logger.info('Workflow finished')
            self.done.set()
            return

        # Submit the next iteration of simulations
        self.logger.info('Submitting next iteration of simulations')
        for sim in self.ensemble.current_sims:
            self.submit_task('simulation', sim, keep_inputs=True)
