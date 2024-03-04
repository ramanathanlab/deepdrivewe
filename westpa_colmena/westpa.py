"""DeepDriveMD WESTPA example."""
from __future__ import annotations

import itertools
import logging
import time
from argparse import ArgumentParser
from functools import partial
from functools import update_wrapper
from pathlib import Path
from typing import Any

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import event_responder
from colmena.thinker import result_processor
from proxystore.store import register_store
from proxystore.store.file import FileStore

from westpa_colmena.api import (  # InferenceCountDoneCallback,
    DeepDriveMDSettings,
)
from westpa_colmena.api import (  # InferenceCountDoneCallback,
    SimulationCountDoneCallback,
)
from westpa_colmena.api import (  # InferenceCountDoneCallback,
    TimeoutDoneCallback,
)
from westpa_colmena.apps.amber_simulation import SimulationResult
from westpa_colmena.parsl import ComputeSettingsTypes


def run_simulation(
    output_dir: Path,
    amber_exe: Path,
    md_input_file: Path,
    prmtop_file: Path,
    checkpoint_file: Path,
    reference_pdb_file: Path,
    cpp_traj_exe: Path,
) -> SimulationResult:
    """Run a simulation and return the pcoord and coordinates."""
    from westpa_colmena.apps.amber_simulation import AmberSimulation
    from westpa_colmena.apps.amber_simulation import CppTrajAnalyzer
    from westpa_colmena.apps.amber_simulation import SimulationResult

    # from proxystore.proxy import Proxy
    # from proxystore.factory import SimpleFactory
    # from westpa_colmena.apps.amber_simulation import SimulationResult as Result
    # class SimulationResult(Result):
    #     def __post_init__(self) -> None:
    #         factory = SimpleFactory(self.coords)
    #         self.coords = Proxy(factory)

    # First run the simulation
    simulation = AmberSimulation(
        amber_exe,
        md_input_file,
        prmtop_file,
        output_dir,
        checkpoint_file,
    )

    # Run the simulation
    simulation.run()

    # Then run cpptraj to get the pcoord and coordinates
    analyzer = CppTrajAnalyzer(
        cpp_traj_exe=cpp_traj_exe,
        reference_pdb_file=reference_pdb_file,
    )
    pcoord = analyzer.get_pcoords(simulation)
    coords = analyzer.get_coords(simulation)

    result = SimulationResult(
        pcoord=pcoord,
        coords=coords,
        restart_file=simulation.restart_file,
        parent_restart_file=checkpoint_file,
    )

    return result


def run_train(input_data: Any) -> None:
    ...


def run_inference(input_data: list[SimulationResult]) -> None:
    ...


class DeepDriveWESTPA(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        simulation_input_dir: Path,
        num_workers: int,
        simulations_per_train: int,
    ) -> None:
        """
        Parameters
        ----------
        queue:
            Queue used to communicate with the task server
        result_dir:
            Directory in which to store outputs
        simulation_input_dir:
            Directory with subdirectories each storing initial simulation start files.
        num_workers:
            Number of workers available for executing simulations, training,
            and inference tasks. One shared worker is reserved for training
            inference task, the rest (num_workers - 1) go to simulation.
        """
        super().__init__(queue)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.num_workers = num_workers

        # Collect initial simulation directories, assumes they are in nested subdirectories
        self.simulation_input_dirs = itertools.cycle(
            filter(lambda p: p.is_dir(), simulation_input_dir.glob('*')),
        )

        # Keep track of the workflow state
        self.simulations_completed = 0

        # Number of times a given task has been submitted
        self.task_counter: defaultdict[str, int] = defaultdict(int)
        self.done_callbacks = done_callbacks

        # Communicate information between agents
        self.simulation_govenor = Semaphore()
        self.run_training = Event()
        self.run_inference = Event()

        # Custom data structures
        self.train_input: list[SimulationResult] = []
        self.inference_input: list[SimulationResult] = []
        # Always want to run inference on all the simulations in the batch
        self.simulations_per_train = simulations_per_train
        self.simulations_per_inference = self.num_workers - 1

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f'{topic}.json', 'a') as f:
            print(result.json(exclude={'inputs', 'value'}), file=f)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            keep_inputs=False,
        )
        self.task_counter[topic] += 1

    @agent  # type: ignore[misc]
    def main_loop(self) -> None:
        while not self.done.is_set():
            for callback in self.done_callbacks:
                if callback.workflow_finished(self):
                    self.logger.info('Exiting DeepDriveMD')
                    self.done.set()
                    return
            time.sleep(1)

    @agent(startup=True)  # type: ignore[misc]
    def start_simulations(self) -> None:
        """Launch a first batch of simulations"""
        # Save one worker for train/inference tasks
        for _ in range(self.num_workers - 1):
            self.simulate()

    @result_processor(topic='simulation')  # type: ignore[misc]
    def process_simulation_result(self, result: Result) -> None:
        """Receive a training result and then submit new tasks

        Will always submit a new simulation tasks and an inference or training
        if the :meth:`handle_simulation_output` sets the appropriate flags.
        """
        # Log simulation job results
        self.log_result(result, 'simulation')
        if not result.success:
            # TODO (wardlt): Should we submit a new simulation if one fails?
            # (braceal): Yes, I think so. I think we can move this check to after submit_task()
            self.logger.warning('Bad simulation result')
            return

        self.simulations_completed += 1

        # NOTE: We had to remove the simulation submit here to implement simulation batching

        # This function is running an implicit while-true loop
        # we need to break out if the done flag has been sent,
        # otherwise it will submit a new simulation.
        if self.done.is_set():
            return

        # Result should be used to train the model and infer new restart points
        self.handle_simulation_output(result.value)

    # TODO (wardlt): We can have this event_responder allocate resources away from simulation if desired.
    @event_responder(event_name='run_training')  # type: ignore[misc]
    def perform_training(self) -> None:
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

    # TODO (wardlt): We can have this event_responder allocate resources away from simulation if desired.
    @event_responder(event_name='run_inference')  # type: ignore[misc]
    def perform_inference(self) -> None:
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

    def simulate(self) -> None:
        """Start a simulation task.

        Must call :meth:`submit_task` with ``topic='simulation'``
        """
        ...

    def train(self) -> None:
        """Start a training task.

        Must call :meth:`submit_task` with ``topic='train'``
        """
        ...

    def inference(self) -> None:
        """Start an inference task

        Must call a :meth:`submit_task` with ``topic='infer'``
        """
        # Inference must wait for a trained model to be available
        while not self.model_weights_available:
            time.sleep(1)

        self.submit_task('inference', self.inference_input)
        self.inference_input = []  # Clear batched data
        self.logger.info('processed inference result')

    def handle_simulation_output(self, output: SimulationResult) -> None:
        """Stores a simulation output in the training set and define new inference tasks

        Should call ``self.run_training.set()`` and/or ``self.run_inference.set()``

        Parameters
        ----------
        output:
            Output to be processed
        """
        # Collect simulation results
        self.train_input.append(output)
        self.inference_input.append(output)
        # Since we are not clearing the train/inference inputs, the length will be the same
        num_sims = len(self.train_input)

        if num_sims and (num_sims % self.simulations_per_train == 0):
            self.run_training.set()

        # TODO: simulations_per_inference should be the same as the number of simulation workers
        if num_sims and (num_sims % self.simulations_per_inference == 0):
            self.run_inference.set()

    def handle_train_output(self, output: Any) -> None:
        """Use the output from a training run to update the model"""
        ...

    def handle_inference_output(self, output: Any) -> None:
        """Use the output from an inference run to update the list of available simulations"""
        # Run the next batch simulations
        for sim_input in output.simulation_inputs:
            self.submit_task('simulation', sim_input)


class ExperimentSettings(DeepDriveMDSettings):
    """Provide a YAML interface to configure the experiment."""

    compute_settings: ComputeSettingsTypes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / 'params.yaml')
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name='file', store_dir=str(cfg.run_dir / 'proxy-store'))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method='pickle',
        topics=['simulation', 'train', 'inference'],
        proxystore_name='file',
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(
        cfg.run_dir / 'run-info',
    )

    # Assign constant settings to each task function
    my_run_simulation = partial(run_simulation)
    my_run_train = partial(run_train)
    my_run_inference = partial(run_inference)
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_train, run_train)
    update_wrapper(my_run_inference, run_inference)

    doer = ParslTaskServer(
        [my_run_simulation, my_run_train, my_run_inference],
        queues,
        parsl_config,
    )

    thinker = DeepDriveWESTPA(
        queue=queues,
        result_dir=cfg.run_dir / 'result',
        simulation_input_dir=cfg.simulation_input_dir,
        num_workers=cfg.num_workers,
        simulations_per_train=cfg.simulations_per_train,
        simulations_per_inference=cfg.simulations_per_inference,
        done_callbacks=[
            # InferenceCountDoneCallback(2),  # Testing
            SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
    )
    logging.info('Created the task server and task generator')

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    store.close()
