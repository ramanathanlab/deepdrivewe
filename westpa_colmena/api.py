"""API for running workflows with Colmena."""

from __future__ import annotations

import json
import time
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from threading import Event
from threading import Semaphore
from typing import Any
from typing import TypeVar
from typing import Union

import yaml  # type: ignore[import-untyped]
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import event_responder
from colmena.thinker import result_processor
from pydantic import BaseModel as _BaseModel

T = TypeVar('T')

PathLike = Union[str, Path]


class BaseModel(_BaseModel):
    """Provide an easy interface to read/write YAML files."""

    def dump_yaml(self, filename: PathLike) -> None:
        """Dump settings to a YAML file."""
        with open(filename, mode='w') as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[T], filename: PathLike) -> T:
        """Load settings from a YAML file."""
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class DoneCallback(ABC):
    """Done callback interface."""

    @abstractmethod
    def workflow_finished(self, workflow: DeepDriveMDWorkflow) -> bool:
        """Return True, if workflow should terminate."""
        ...


class SimulationCountDoneCallback(DoneCallback):
    """Simulation count done callback.

    Exit from DeepDriveMD after a certain number of simulations have
    finished.
    """

    def __init__(self, total_simulations: int) -> None:
        """Initialize the simulation count done callback.

        Parameters
        ----------
        total_simulations : int
            Total number of simulations to run.
        """
        self.total_simulations = total_simulations

    def workflow_finished(self, workflow: DeepDriveMDWorkflow) -> bool:
        """Return True, if workflow should terminate."""
        return workflow.task_counter['simulation'] >= self.total_simulations


class InferenceCountDoneCallback(DoneCallback):
    """Inference count done callback.

    Exit from DeepDriveMD after a certain number of inference tasks have
    finished.
    """

    def __init__(self, total_inferences: int) -> None:
        """Initialize the inference count done callback.

        Parameters
        ----------
        total_inferences : int
            Total number of inference tasks to run.
        """
        self.total_inferences = total_inferences

    def workflow_finished(self, workflow: DeepDriveMDWorkflow) -> bool:
        """Return True, if workflow should terminate."""
        return workflow.task_counter['inference'] >= self.total_inferences


class ResultLogger:
    """Logger for results."""

    def __init__(self, result_dir: Path) -> None:
        """Initialize the result logger.

        Parameters
        ----------
        result_dir: Path
            Directory in which to store outputs
        """
        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir

        # Number of times a given task has been submitted
        self.task_counter: defaultdict[str, int] = defaultdict(int)

    def log(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        # Increment the task counter
        self.task_counter[topic] += 1

        # Write the result to a jsonl file
        with open(self.result_dir / f'{topic}.json', 'a') as f:
            print(result.json(exclude={'inputs', 'value'}), file=f)


class DeepDriveMDWorkflow(BaseThinker):
    """Base class for DeepDriveMD workflows."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        done_callbacks: list[DoneCallback] | None,
        async_simulation: bool = False,
    ) -> None:
        """Initialize the DeepDriveMD workflow.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server
        result_dir: Path
            Directory in which to store outputs
        done_callbacks: list[DoneCallback]
            Callbacks that can trigger a run to end.
        async_simulation: bool
            If True, the simulation will be run asynchronously, otherwise
            it will be run synchronously.
        """
        super().__init__(queue)

        self.async_simulation = async_simulation
        self.result_logger = ResultLogger(result_dir)

        # Number of times a given task has been submitted
        self.task_counter: defaultdict[str, int] = defaultdict(int)
        self.done_callbacks = done_callbacks or []

        # Communicate information between agents
        self.simulation_govenor = Semaphore()
        self.run_training = Event()
        self.run_inference = Event()

    def log_result(self, result: Result, topic: str) -> None:
        """Log a result to the result logger."""
        # Log the jsonl result
        self.result_logger.log(result, topic)

        # Increment the task counter
        self.task_counter[topic] += 1

    def submit_task(self, topic: str, *inputs: Any) -> None:
        """Submit a task to the task server."""
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            keep_inputs=False,
        )

    @agent
    def main_loop(self) -> None:
        """Run main loop for the DeepDriveMD workflow."""
        while not self.done.is_set():
            for callback in self.done_callbacks:
                if callback.workflow_finished(self):
                    self.logger.info('Exiting DeepDriveMD')
                    self.done.set()
                    return
            time.sleep(1)

    @result_processor(topic='simulation')
    def process_simulation_result(self, result: Result) -> None:
        """Process a simulation result.

        Will always submit a new simulation tasks and an inference or training
        if the :meth:`handle_simulation_output` sets the appropriate flags.
        """
        # Log simulation job results
        self.log_result(result, 'simulation')
        if not result.success:
            # TODO (wardlt): Should we submit a new simulation if one fails?
            # (braceal): Yes, I think so. I think we can move this check to
            # after submit_task()
            self.logger.warning('Bad simulation result')
            return

        # This function is running an implicit while-true loop
        # we need to break out if the done flag has been sent,
        # otherwise it will submit a new simulation.
        if self.done.is_set():
            return

        # Submit another simulation as soon as the previous one finishes
        # to keep utilization high
        if self.async_simulation:
            self.simulate()

        # Result should be used to train the model and infer new restart points
        self.handle_simulation_output(result.value)

    @event_responder(event_name='run_training')
    def perform_training(self) -> None:
        """Perform a training task."""
        self.logger.info('Started training process')

        # Send in a training task
        self.train()

        # Wait for the result to complete
        result: Result = self.queues.get_result(topic='train')
        self.logger.info('Received training result')

        self.log_result(result, 'train')
        if not result.success:
            self.logger.warning('Bad train result')
            return

        # Process the training output
        self.handle_train_output(result.value)
        self.logger.info('Training process is complete')

    @event_responder(event_name='run_inference')
    def perform_inference(self) -> None:
        """Perform an inference task."""
        self.logger.info('Started inference process')

        # Send in an inference task
        self.inference()

        # Wait for the result to complete
        result: Result = self.queues.get_result(topic='inference')
        self.logger.info('Received inference result')

        self.log_result(result, 'inference')
        if not result.success:
            self.logger.warning('Bad inference result')
            return

        # Process the inference output
        self.handle_inference_output(result.value)
        self.logger.info('Inference process is complete')

    @abstractmethod
    def simulate(self) -> None:
        """Start simulation task(s).

        Must call :meth:`submit_task` with ``topic='simulation'``
        """
        ...

    @abstractmethod
    def train(self) -> None:
        """Start a training task.

        Must call :meth:`submit_task` with ``topic='train'``
        """
        ...

    @abstractmethod
    def inference(self) -> None:
        """Start an inference task.

        Must call a :meth:`submit_task` with ``topic='infer'``
        """
        ...

    @abstractmethod
    def handle_simulation_output(self, output: Any) -> None:
        """Handle a simulation output.

        Stores a simulation output in the training set and define
        new inference tasks. Should call ``self.run_training.set()``
        and/or ``self.run_inference.set()``

        Parameters
        ----------
        output:
            Output to be processed
        """
        ...

    @abstractmethod
    def handle_train_output(self, output: Any) -> None:
        """Handle a training output.

        Use the output from a training run to update the model.
        """
        ...

    @abstractmethod
    def handle_inference_output(self, output: Any) -> None:
        """Handle an inference output.

        Use the output from an inference run to update the list of
        available simulations.
        """
        ...
