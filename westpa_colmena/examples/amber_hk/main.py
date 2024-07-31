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
from proxystore.connectors.file import FileConnector
from proxystore.store import register_store
from proxystore.store import Store
from pydantic import Field
from pydantic import validator

from westpa_colmena.api import BaseModel
from westpa_colmena.api import DeepDriveMDWorkflow
from westpa_colmena.api import DoneCallback
from westpa_colmena.api import InferenceCountDoneCallback
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import IterationMetadata
from westpa_colmena.ensemble import SimMetadata
from westpa_colmena.ensemble import TargetState
from westpa_colmena.ensemble import WeightedEnsemble
from westpa_colmena.examples.amber_hk.inference import InferenceConfig
from westpa_colmena.examples.amber_hk.inference import run_inference
from westpa_colmena.examples.amber_hk.simulate import run_simulation
from westpa_colmena.examples.amber_hk.simulate import SimResult
from westpa_colmena.examples.amber_hk.simulate import SimulationConfig
from westpa_colmena.io import WestpaH5File
from westpa_colmena.parsl import ComputeSettingsTypes
from westpa_colmena.simulation.amber import run_cpptraj

# TODO: Next steps:
# (1) Reproduce a binning example to see if our system is working.
# (1.1) TODO: Test changes. We can set the target state closer to the basis
# state to see if the recycle logic is working.
# (2) Pack outputs into HDF5 for westpa analysis.
# (3) Test the resampler and weighted ensemble logic using ntl9.
# (4) Create a pytest for the WESTPA thinker.
# (5) Implement a cleaner thinker backend
# (6) Send cpptraj output to a separate log file to avoid polluting the main

# TODO: Right now if any errors occur in the simulations, then it will
# stop the entire workflow since no inference tasks will be submitted.
# We should resubmit failed workers once and otherwise raise an error and exit.


class DeepDriveWESTPA(DeepDriveMDWorkflow):
    """A WESTPA thinker for DeepDriveMD."""

    def __init__(  # noqa: PLR0913
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsemble,
        westpa_h5file: WestpaH5File,
        basis_states: BasisStates,
        target_states: list[TargetState],
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
        self.basis_states = basis_states
        self.target_states = target_states
        self.westpa_h5file = westpa_h5file

        # TODO: Update the inputs to support model weights, etc
        # Custom data structures
        # self.train_input: list[SimResult] = []
        self.inference_input: list[SimResult] = []

        # Make sure there has been at least one training task
        # complete before running inference
        # self.model_weights_available = False

    @agent(startup=True)
    def start_simulations(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        self.simulate()

    def simulate(self) -> None:
        """Start simulation task(s).

        Must call :meth:`submit_task` with ``topic='simulation'``.
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

    def handle_simulation_output(self, output: SimResult) -> None:
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
        if len(self.inference_input) % num_simulations == 0:
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
        output: tuple[list[SimMetadata], list[SimMetadata], IterationMetadata],
    ) -> None:
        """Handle the output of an inference run.

        Use the output from an inference run to update the list of
        available simulations.
        """
        # Unpack the output
        cur_sims, next_sims, iter_data = output

        # Update the weighted ensemble with the next iteration
        self.ensemble.advance_iteration(next_iteration=next_sims)

        # Submit the next iteration of simulations
        self.simulate()

        # Log the results to the HDF5 file
        # TODO: Update the basis states and target states (right now
        # it assumes they are static, but once we add these to the iteration
        # metadata, they can be returned neatly from the inference function.)
        # TODO: This requires making BasisStates a pydantic BaseModel.
        self.westpa_h5file.append(
            cur_iteration=cur_sims,
            basis_states=self.basis_states,
            target_states=self.target_states,
            metadata=iter_data,
        )

        # Log the current iteration
        self.logger.info(
            f'Current iteration: {len(self.ensemble.simulations)}',
        )


class MyBasisStates(BasisStates):
    """Custom basis state initialization."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the basis states."""
        self.sim_config = sim_config
        super().__init__(*args, **kwargs)

    def init_basis_pcoord(self, basis_file: Path) -> list[float]:
        """Initialize the basis state parent coordinates."""
        # Create the cpptraj command file
        command = (
            f'parm {self.sim_config.amber_config.top_file} \n'
            f'trajin {basis_file}\n'
            f'reference {self.sim_config.reference_file} [reference] \n'
            'distance na-cl :1@Na+ :2@Cl- out {output_file} \n'
            'go'
        )
        return run_cpptraj(command)


class ExperimentSettings(BaseModel):
    """Provide a YAML interface to configure the experiment."""

    simulation_input_dir: Path = Field(
        description='Nested directory storing initial simulation start files, '
        'e.g. pdb_dir/system1/, pdb_dir/system2/, ..., where system<i> might '
        'store PDB files, topology files, etc needed to start the simulation '
        'application.',
    )
    output_dir: Path = Field(
        description='Directory in which to store the results.',
    )
    basis_state_ext: str = Field(
        default='.ncrst',
        description='Extension for the basis states.',
    )
    initial_ensemble_members: int = Field(
        description='Number of simulations to start the weighted ensemble.',
    )
    num_iterations: int = Field(
        description='Number of iterations to run the weighted ensemble.',
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
    target_states: list[TargetState] = Field(
        description='The target threshold for the progress coordinate to be'
        ' considered in the target state.',
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

    # Make the store
    store = Store(
        name='file-store',
        connector=FileConnector(store_dir=str(cfg.output_dir / 'proxy-store')),
    )

    # TODO: This won't be needed in the next colmena release
    # Register the store
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method='pickle',
        topics=['simulation', 'inference'],
        proxystore_name='file-store',
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(
        cfg.output_dir / 'run-info',
    )

    # Initialize the basis states
    basis_states = MyBasisStates(
        sim_config=cfg.simulation_config,
        initial_ensemble_members=cfg.initial_ensemble_members,
        simulation_input_dir=cfg.simulation_input_dir,
        basis_state_ext=cfg.basis_state_ext,
    )

    # Print the basis states
    logging.info(f'Basis states: {basis_states.basis_states}')

    # Initialize the weighted ensemble
    ensemble = WeightedEnsemble(
        basis_states=basis_states,
        checkpoint_dir=cfg.output_dir / 'checkpoint',
        resume_checkpoint=cfg.resume_checkpoint,
    )

    # Assign constant settings to each task function
    my_run_simulation = partial(
        run_simulation,
        config=cfg.simulation_config,
        output_dir=cfg.output_dir / 'simulation',
    )
    my_run_inference = partial(
        run_inference,
        basis_states=basis_states,
        target_states=cfg.target_states,
        config=cfg.inference_config,
    )
    update_wrapper(my_run_simulation, run_simulation)
    update_wrapper(my_run_inference, run_inference)

    # Create the task server
    doer = ParslTaskServer(
        [my_run_simulation, my_run_inference],
        queues,
        parsl_config,
    )

    # Add a callbacks to decide when to stop the thinker
    done_callbacks = [
        InferenceCountDoneCallback(total_inferences=cfg.num_iterations),
    ]

    # Create the HDF5 file for WESTPA
    # TODO: Unify the westpa hdf5 file with the checkpoint logic.
    westpa_h5file = WestpaH5File(cfg.output_dir / 'west.h5')

    # Create the workflow thinker
    thinker = DeepDriveWESTPA(
        queue=queues,
        result_dir=cfg.output_dir / 'result',
        ensemble=ensemble,
        westpa_h5file=westpa_h5file,
        basis_states=basis_states,
        target_states=cfg.target_states,
        done_callbacks=done_callbacks,  # type: ignore[arg-type]
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
