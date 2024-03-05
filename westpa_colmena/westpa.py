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

from colmena.queue import ColmenaQueues
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from proxystore.store import register_store
from proxystore.store.file import FileStore

from westpa_colmena.api import DeepDriveMDSettings
from westpa_colmena.api import DeepDriveMDWorkflow
from westpa_colmena.api import DoneCallback
from westpa_colmena.api import SimulationCountDoneCallback
from westpa_colmena.api import TimeoutDoneCallback
from westpa_colmena.apps.amber_simulation import SimulationResult
from westpa_colmena.apps.pcoord_inference import WeightedEnsemble
from westpa_colmena.parsl import ComputeSettingsTypes

# TODO: We need to send a random seed for amber simulations
# TODO: It may good to implement the abstract batched simulation pattern
#       within DeepDriveMD to do ablations and make the current westpa
#       thinker implementation more elegant


def run_simulation(  # noqa: PLR0913
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
    """Train a model on the input data."""
    ...


def run_inference(input_data: list[SimulationResult]) -> None:
    """Run inference on the input data."""
    ...


class DeepDriveWESTPA(DeepDriveMDWorkflow):
    """A WESTPA thinker for DeepDriveMD."""

    def __init__(  # noqa: PLR0913
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        simulation_input_dir: Path,
        num_workers: int,
        done_callbacks: list[DoneCallback],
        ensemble_members: int,
    ) -> None:
        """Initialize the DeepDriveWESTPA thinker.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server
        result_dir: Path
            Directory in which to store outputs
        simulation_input_dir:
            Directory with subdirectories each storing initial simulation
            start files.
        num_workers: int
            Number of workers available for executing simulations, training,
            and inference tasks. One shared worker is reserved for training
            inference task, the rest (num_workers - 1) go to simulation.
        done_callbacks: list[DoneCallback]
            List of callbacks to determine when the thinker should stop.
        ensemble_members: int
            The number of simulations to start the weighted ensemble with.
        """
        super().__init__(
            queue,
            result_dir,
            simulation_input_dir,
            num_workers,
            done_callbacks,
            async_simulation=False,
        )

        # TODO: Check that this logic is correct
        # Get the basis states by globbing the simulation input directories
        basis_states: list[Path] = [  # type: ignore[misc]
            p.glob('.ncrst')
            for p in itertools.islice(
                self.simulation_input_dirs,
                ensemble_members,
            )
        ]

        # Initialize the weighted ensemble
        self.weighted_ensemble = WeightedEnsemble(
            basis_states=basis_states,
            ensemble_members=ensemble_members,
        )

        # TODO: Update the inputs to support model weights, etc
        # Custom data structures
        self.train_input: list[SimulationResult] = []
        self.inference_input: list[SimulationResult] = []

        self.current_idx = 0
        self.ensemble_members = ensemble_members

        # Make sure there has been at least one training task
        # complete before running inference
        self.model_weights_available = False

    def simulate(self) -> None:
        """Start a simulation task.

        Must call :meth:`submit_task` with ``topic='simulation'``
        """
        simulations = self.weighted_ensemble.current_iteration
        self.submit_task('simulation', simulations[self.current_idx])
        self.current_idx += 1

    def train(self) -> None:
        """Start a training task.

        Must call :meth:`submit_task` with ``topic='train'``.
        """
        self.submit_task('train', self.train_input)

    def inference(self) -> None:
        """Start an inference task.

        Must call a :meth:`submit_task` with ``topic='infer'``.
        """
        # Inference must wait for a trained model to be available
        while not self.model_weights_available:
            time.sleep(1)

        self.submit_task('inference', self.inference_input)
        self.inference_input = []  # Clear batched data
        self.logger.info('processed inference result')

    def handle_simulation_output(self, output: SimulationResult) -> None:
        """Handle the output of a simulation.

        Stores a simulation output in the training set and define new
        inference tasks Should call ``self.run_training.set()``
        and/or ``self.run_inference.set()``.

        Parameters
        ----------
        output:
            Output to be processed
        """
        # Collect simulation results
        self.train_input.append(output)
        self.inference_input.append(output)

        # Since we are not clearing the train/inference inputs, the
        # length will be the same as the ensemble members
        if self.ensemble_members % len(self.train_input) == 0:
            self.run_training.set()
            self.run_inference.set()

    def handle_train_output(self, output: Any) -> None:
        """Use the output from a training run to update the model."""
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True
        self.logger.info(
            f'Updated model_weight_path to: {output.model_weight_path}',
        )

    def handle_inference_output(self, output: Any) -> None:
        """Handle the output of an inference run.

        Use the output from an inference run to update the list of
        available simulations.
        """
        # TODO: Figure out exactly what the output needs to contain
        # In fact, the inference should just return the binning output
        to_split, to_merge = self.binner.bin(output)

        self.weighted_ensemble.advance_iteration(to_split, to_merge)

        # Reset the current simulation index for the next iteration
        self.current_idx = 0

        # Run the next batch simulations
        for _ in range(self.num_workers - 1):
            self.simulate()


class ExperimentSettings(DeepDriveMDSettings):
    """Provide a YAML interface to configure the experiment."""

    ensemble_members: int
    """Number of simulations to start the weighted ensemble with."""
    compute_settings: ComputeSettingsTypes
    """Settings for the compute environment."""


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
        done_callbacks=[
            # InferenceCountDoneCallback(2),  # Testing
            SimulationCountDoneCallback(cfg.num_total_simulations),
            TimeoutDoneCallback(cfg.duration_sec),
        ],
        ensemble_members=cfg.ensemble_members,
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
