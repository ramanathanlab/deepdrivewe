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
    """A thinker for the WESTPA workflow."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
        max_retries: int = 2,
    ) -> None:
        """Initialize the WESTPA workflow thinker.

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
        max_retries: int
            Number of times to retry a task if it fails (default to 2).
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations
        self.max_retries = max_retries
        self.result_logger = ResultLogger(result_dir)

        # Store the simulation output (the input of the inference task)
        self.sim_output: list[Any] = []

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

        # Collect simulation results
        self.sim_output.append(result.value)

        if len(self.sim_output) == len(self.ensemble.next_sims):
            self.submit_task('inference', self.sim_output)
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
