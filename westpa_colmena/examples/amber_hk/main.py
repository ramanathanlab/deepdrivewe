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

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import agent
from colmena.thinker import BaseThinker
from colmena.thinker import result_processor
from proxystore.connectors.file import FileConnector
from proxystore.store import register_store
from proxystore.store import Store
from pydantic import Field
from pydantic import validator

from westpa_colmena.api import BaseModel
from westpa_colmena.api import ResultLogger
from westpa_colmena.checkpoint import EnsembleCheckpointer
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import TargetState
from westpa_colmena.ensemble import WeightedEnsembleV2
from westpa_colmena.examples.amber_hk.inference import InferenceConfig
from westpa_colmena.examples.amber_hk.inference import run_inference
from westpa_colmena.examples.amber_hk.simulate import run_simulation
from westpa_colmena.examples.amber_hk.simulate import SimResult
from westpa_colmena.examples.amber_hk.simulate import SimulationConfig
from westpa_colmena.parsl import ComputeSettingsTypes
from westpa_colmena.simulation.amber import run_cpptraj

# TODO: Next steps:
# (1) Test the resampler and weighted ensemble logic using ntl9.
# (2) Create a pytest for the WESTPA thinker.
# (3) Send cpptraj output to a separate log file to avoid polluting the main
# (4) Support checkpointing for the WESTPA thinker
# (5) Forward some of the imports and unify api and ensemble imports.
# (6) Call package ddwe.
# (7) Address west.cfg file requirement for WESTPA analysis tools.

# TODO: Right now if any errors occur in the simulations, then it will
# stop the entire workflow since no inference tasks will be submitted.
# We should resubmit failed workers once and otherwise raise an error and exit.

# TODO: It looks like this thinker implements all the base WESTPA cases.
#       Maybe we should move it to the API.

# TODO: See TODOs in process_inference_result for pydantic refactor.

# TODO: We should look at the h5 for tstate and bstate info for checkpointing +
# to accounnt for changes in the h5.

# TODO: Next step: see ensemble.py to finish implementing WeightedEnsembleV2.


class SynchronousDDWE(BaseThinker):
    """A synchronous DDWE thinker."""

    def __init__(
        self,
        queue: ColmenaQueues,
        result_dir: Path,
        ensemble: WeightedEnsembleV2,
        checkpointer: EnsembleCheckpointer,
        num_iterations: int,
    ) -> None:
        """Initialize the DeepDriveMD workflow.

        Parameters
        ----------
        queue: ColmenaQueues
            Queue used to communicate with the task server
        result_dir: Path
            Directory in which to store outputs
        num_iterations: int
            Number of iterations to run the workflow.
        """
        super().__init__(queue)

        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.num_iterations = num_iterations

        self.inference_input: list[SimResult] = []
        self.result_logger = ResultLogger(result_dir)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        """Submit a task to the task server."""
        self.queues.send_inputs(
            *inputs,
            method=f'run_{topic}',
            topic=topic,
            keep_inputs=False,
        )

    @agent(startup=True)
    def start_simulations(self) -> None:
        """Launch the first iteration of simulations to start the workflow."""
        # Submit the next iteration of simulations
        for sim in self.ensemble.current_sims:
            self.submit_task('simulation', sim)

    @result_processor(topic='simulation')
    def process_simulation_result(self, result: Result) -> None:
        """Process a simulation result."""
        # Log simulation job results
        self.result_logger.log(result, topic='simulation')

        if not result.success:
            # TODO (wardlt): Should we submit a new simulation if one fails?
            # (braceal): Yes, I think so. I think we can move this check to
            # after submit_task()
            self.logger.warning('Bad simulation result')
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
            self.logger.warning('Bad inference result')
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

        # TODO: Update the basis states and target states (right now
        # it assumes they are static, but once we add these to the iteration
        # metadata, they can be returned neatly from the inference function.)
        # TODO: This requires making BasisStates a pydantic BaseModel.
        # TODO: The ensemble can also be a pydantic class to facilitate
        # easy logging to json for readable checkpoints.
        # TODO: We should consider whether to make the westpa h5 file a
        # pydantic class in order to serialize the metadata with the checkpoint
        # file. However, we might want to think about the case where a user
        # wants to resume a checkpoint in a different directory. Is this
        # a supported case?

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
            self.submit_task('simulation', sim)


class MyBasisStates(BasisStates):
    """Custom basis state initialization."""

    top_file: Path = Field(
        description='Topology file for the cpptraj command.',
    )
    reference_file: Path = Field(
        description='Reference file for the cpptraj command.',
    )

    def init_basis_pcoord(self, basis_file: Path) -> list[float]:
        """Initialize the basis state parent coordinates."""
        # Create the cpptraj command file
        command = (
            f'parm {self.top_file} \n'
            f'trajin {basis_file}\n'
            f'reference {self.reference_file} [reference] \n'
            'distance na-cl :1@Na+ :2@Cl- out {output_file} \n'
            'go'
        )
        return run_cpptraj(command)


class ExperimentSettings(BaseModel):
    """Provide a YAML interface to configure the experiment."""

    output_dir: Path = Field(
        description='Directory in which to store the results.',
    )
    num_iterations: int = Field(
        ge=1,
        description='Number of iterations to run the weighted ensemble.',
    )
    basis_states: MyBasisStates = Field(
        description='The basis states for the weighted ensemble.',
    )
    target_states: list[TargetState] = Field(
        description='The target threshold for the progress coordinate to be'
        ' considered in the target state.',
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

    # Create the checkpoint manager
    checkpointer = EnsembleCheckpointer(output_dir=cfg.output_dir)

    # Check if a checkpoint exists
    checkpoint = checkpointer.latest_checkpoint()

    if checkpoint is None:
        # Initialize the weighted ensemble
        ensemble = WeightedEnsembleV2(
            basis_states=cfg.basis_states,
            target_states=cfg.target_states,
        )

        # TODO: temp hack to load the basis states
        # Load the basis states
        ensemble.basis_states.load_basis_states()
        from copy import deepcopy

        # Initialize the simulations with the basis states
        ensemble.simulations = [deepcopy(ensemble.basis_states.basis_states)]
    else:
        # Load the ensemble from a checkpoint if it exists
        ensemble = checkpointer.load(checkpoint)
        logging.info(f'Loaded ensemble from checkpoint {checkpoint}')

    # Print the input states
    logging.info(f'Basis states: {ensemble.basis_states}')
    logging.info(f'Target states: {ensemble.target_states}')

    # Assign constant settings to each task function
    my_run_simulation = partial(
        run_simulation,
        config=cfg.simulation_config,
        output_dir=cfg.output_dir / 'simulation',
    )
    my_run_inference = partial(
        run_inference,
        basis_states=ensemble.basis_states,
        target_states=ensemble.target_states,
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

    # Create the workflow thinker
    thinker = SynchronousDDWE(
        queue=queues,
        result_dir=cfg.output_dir / 'result',
        ensemble=ensemble,
        checkpointer=checkpointer,
        num_iterations=cfg.num_iterations,
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
