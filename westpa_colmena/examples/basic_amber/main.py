"""DeepDriveMD WESTPA example.

Adapted from:
https://github.com/westpa/westpa_tutorials/tree/main/additional_tutorials/basic_nacl_amber
"""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from functools import partial
from functools import update_wrapper
from pathlib import Path
from typing import Any

from colmena.queue import ColmenaQueues
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import agent
from proxystore.store import register_store
from proxystore.store.file import FileStore
from pydantic import Field
from pydantic import validator

from westpa_colmena.api import BaseModel
from westpa_colmena.api import DeepDriveMDWorkflow
from westpa_colmena.api import DoneCallback
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import SimulationMetadata
from westpa_colmena.ensemble import WeightedEnsemble
from westpa_colmena.examples.basic_amber.inference import InferenceConfig
from westpa_colmena.examples.basic_amber.inference import run_inference
from westpa_colmena.examples.basic_amber.simulate import run_simulation
from westpa_colmena.examples.basic_amber.simulate import SimulationConfig
from westpa_colmena.examples.basic_amber.simulate import SimulationResult
from westpa_colmena.parsl import ComputeSettingsTypes

# TODO: We need to send a random seed for amber simulations

# TODO: Next steps:
# (1) Implement the run_inference function using the resampler.
# (2) Define the input arguments and return results for the tasks.
#       - Simulation results as a field within the metadata would do it.
#       - Think about how to store simulation results (such as coordinates)
#         in a way that can be proxied so that the thinker does not
#         implicitly load all the data into memory.
# (3) Test the resampler and weighted ensemble logic in pytest.
# (4) Create a pytest for the WESTPA thinker.


class DeepDriveWESTPA(DeepDriveMDWorkflow):
    """A WESTPA thinker for DeepDriveMD."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        done_callbacks: list[DoneCallback] | None = None,
    ) -> None:
        """Initialize the DeepDriveWESTPA thinker.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server
        result_dir: Path
            Directory in which to store outputs
        ensemble: WeightedEnsemble
            Weighted ensemble object to manage the simulations and weights.
        done_callbacks: list[DoneCallback] | None
            List of callbacks to determine when the thinker should stop.
        """
        super().__init__(
            queue,
            result_dir,
            done_callbacks,
            async_simulation=False,
        )

        self.ensemble = ensemble

        # TODO: Update the inputs to support model weights, etc
        # Custom data structures
        # self.train_input: list[SimulationResult] = []
        self.inference_input: list[SimulationResult] = []

        # Make sure there has been at least one training task
        # complete before running inference
        # self.model_weights_available = False

    @agent(startup=True)
    def start_simulations(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        self.simulate()

    def simulate(self) -> None:
        """Start simulation task(s).

        Must call :meth:`submit_task` with ``topic='simulation'``
        """
        # Submit the next iteration of simulations
        for sim in self.ensemble.current_iteration:
            self.submit_task('simulation', sim)

    def train(self) -> None:
        """Start a training task.

        Must call :meth:`submit_task` with ``topic='train'``.
        """
        # self.submit_task('train', self.train_input)
        pass

    def inference(self) -> None:
        """Start an inference task.

        Must call a :meth:`submit_task` with ``topic='infer'``.
        """
        # Inference must wait for a trained model to be available
        # while not self.model_weights_available:
        #     time.sleep(1)

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
        # self.train_input.append(output)
        self.inference_input.append(output)

        # Number of simulations in the current iteration
        num_simulations = len(self.ensemble.current_iteration)

        # Since we are not clearing the train/inference inputs, the
        # length will be the same as the ensemble members
        if num_simulations % len(self.inference_input) == 0:
            # self.run_training.set()
            self.run_inference.set()

    def handle_train_output(self, output: Any) -> None:
        """Use the output from a training run to update the model."""
        # self.inference_input.model_weight_path = output.model_weight_path
        # self.model_weights_available = True
        # self.logger.info(
        #     f'Updated model_weight_path to: {output.model_weight_path}',
        # )
        pass

    def handle_inference_output(
        self,
        output: list[SimulationMetadata],
    ) -> None:
        """Handle the output of an inference run.

        Use the output from an inference run to update the list of
        available simulations.
        """
        # Update the weighted ensemble with the next iteration
        self.ensemble.advance_iteration(next_iteration=output)

        # Submit the next iteration of simulations
        self.simulate()


class ExperimentSettings(BaseModel):
    """Provide a YAML interface to configure the experiment."""

    simulation_input_dir: Path = Field(
        description='Nested directory storing initial simulation start files, '
        'e.g. pdb_dir/system1/, pdb_dir/system2/, ..., where system<i> might '
        'store PDB files, topology files, etc needed to start the simulation '
        'application.',
    )
    basis_state_ext: str = Field(
        default='.ncrst',
        description='Extension for the basis states.',
    )
    ensemble_members: int = Field(
        description='Number of simulations to start the weighted ensemble.',
    )
    output_dir: Path = Field(
        description='Directory in which to store the results.',
    )
    resume_checkpoint: Path | None = Field(
        default=None,
        description='Path to the checkpoint file.',
    )
    simulation_config: SimulationConfig = Field(
        description='Arguments for the simulation.',
    )
    inference_config: InferenceConfig = Field(
        description='Arguments for the inference.',
    )
    compute_settings: ComputeSettingsTypes = Field(
        description='Settings for the compute resources.',
    )

    @validator('output_dir')
    @classmethod
    def mkdir_validator(cls, value: Path) -> Path:
        """Resolve and make the output directory."""
        value = value.resolve()
        value.mkdir(parents=True, exist_ok=True)
        return value


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(cfg.output_dir / 'runtime.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Make the proxy store
    store = FileStore(
        name='file',
        store_dir=str(cfg.output_dir / 'proxy-store'),
    )
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method='pickle',
        topics=['simulation', 'inference'],
        proxystore_name='file',
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(
        cfg.output_dir / 'run-info',
    )

    # Assign constant settings to each task function
    my_run_simulation = partial(
        run_simulation,
        output_dir=cfg.output_dir / 'simulation',
        config=cfg.simulation_config,
    )
    my_run_inference = partial(run_inference, config=cfg.inference_config)
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_inference, run_inference)

    # Create the task server
    doer = ParslTaskServer(
        [my_run_simulation, my_run_inference],
        queues,
        parsl_config,
    )

    # Initialize the basis states
    basis_states = BasisStates(
        ensemble_members=cfg.ensemble_members,
        simulation_input_dir=cfg.simulation_input_dir,
        basis_state_ext=cfg.basis_state_ext,
    )

    # Initialize the weighted ensemble
    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        checkpoint_dir=cfg.output_dir / 'checkpoint',
        resume_checkpoint=cfg.resume_checkpoint,
    )

    thinker = DeepDriveWESTPA(
        queue=queues,
        result_dir=cfg.output_dir / 'result',
        ensemble=ensemble,
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
