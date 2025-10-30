"""Training module."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from deepdrivewe import SimResult
from deepdrivewe import TrainResult
from deepdrivewe import validate_and_resolve_file
from deepdrivewe.ai import ConvolutionalVAE
from deepdrivewe.ai import ConvolutionalVAEConfig
from deepdrivewe.workflows.stream import ProxyStreamConfig
from deepdrivewe.workflows.stream import SIMULATION_TOPIC
from deepdrivewe.workflows.stream import TRAIN_TOPIC


class TrainConfig(BaseModel):
    """Arguments for the training module."""

    config_path: Path = Field(
        description='The path to the model configuration file.',
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description='The path to the model checkpoint file.'
        'Train from scratch by default.',
    )
    stream_items_per_train: int = Field(
        default=1,
        description='The number of items (simulation frames) to train on in '
        'each stream iteration.',
    )
    stream_retrain_interval: int = Field(
        default=1,
        description='The number of stream training iterations between '
        're-initializing and re-training the model.',
    )

    @field_validator('config_path', 'checkpoint_path')
    @classmethod
    def validate_and_resolve_file(cls, value: Path | None) -> Path | None:
        """Validate and resolve the file path."""
        return validate_and_resolve_file(value)


# TODO: We probably need to store a history of old training data
# to retrain the model. Add a config argument to include a cMD run dataset.
# Contact maps: https://github.com/n-frazee/DL-enhancedWE/blob/main/common_files/train.npy


def run_train(
    sim_output: list[SimResult],
    config: TrainConfig,
    output_dir: Path,
    stream_config: ProxyStreamConfig | None = None,
) -> TrainResult:
    """Train the model on the simulation output."""
    # If we are using a stream, run the stream training function
    if stream_config is not None:
        return run_stream_train(
            config=config,
            output_dir=output_dir,
            stream_config=stream_config,
        )

    # Make the output directory
    itetation = sim_output[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the model configuration
    model_config = ConvolutionalVAEConfig.from_yaml(config.config_path)

    # Load the model
    model = ConvolutionalVAE(
        model_config,
        checkpoint_path=config.checkpoint_path,
    )

    # Flatten all contact maps and pcoords from all sims into a single array
    contact_maps = np.array(
        [cm for sim in sim_output for cm in sim.data['contact_maps']],
        dtype=object,
    )
    pcoords = np.concatenate([sim.data['pcoords'] for sim in sim_output])
    pcoords = pcoords.flatten()

    # Fit the model
    checkpoint_path = model.fit(
        x=contact_maps,
        model_dir=output_dir / 'model',
        scalars={'pcoord': pcoords},
    )

    # Return the train result
    result = TrainResult(
        config_path=config.config_path,
        checkpoint_path=checkpoint_path,
    )

    return result


def run_stream_train(
    config: TrainConfig,
    output_dir: Path,
    stream_config: ProxyStreamConfig,
) -> TrainResult:
    """Train the model on the simulation output."""
    # Make the output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream consumer for getting new simulation data
    stream_consumer = stream_config.get_consumer(topic=SIMULATION_TOPIC)
    # Stream producer for sending new trained model weights to the thinker
    stream_producer = stream_config.get_producer(topic=TRAIN_TOPIC)

    # TODO: Decide how much data we want to keep in the re-train history.

    # Loop indefinitely until we get a stop iteration from the stream consumer
    for idx in itertools.count():
        print(f'Train iteration: {idx}', flush=True)
        # If we have reached the retrain interval, re-initialize the trainer
        # NOTE: This always happens on the first iteration
        if idx % config.stream_retrain_interval == 0:
            print(f'Retraining model at iteration: {idx}', flush=True)
            # Load the model configuration
            model_config = ConvolutionalVAEConfig.from_yaml(config.config_path)

            # Load the model
            model = ConvolutionalVAE(
                model_config,
                checkpoint_path=config.checkpoint_path,
            )

        print(
            f'Getting next batch of simulation data at iteration: {idx}',
            flush=True,
        )
        # Get the next batch of simulation data from the stream.
        # Each item is a dictionary with topic keys defined in the simulation
        # module, (e.g. 'contact_maps', 'pcoords', etc.), and values are
        # numpy arrays representing a single frame of data.
        try:
            items = [
                next(stream_consumer)
                for _ in range(config.stream_items_per_train)
            ]
        except StopIteration:
            print(
                f'Reached end of training stream consumer at iteration: {idx}',
                flush=True,
            )
            break

        # Extract the contact maps and rmsd from each simulation
        print(
            f'Got {len(items)} items from stream consumer at iteration: {idx}',
            flush=True,
        )
        cmaps = np.array([x['contact_maps'] for x in items], dtype=object)

        print(f'Contact maps: {cmaps.shape}', flush=True)
        print(f'Contact maps[0]: {cmaps[0]}', flush=True)

        pcoords = np.array([x['pcoords'] for x in items]).reshape(-1, 1)

        print(f'Pcoords: {pcoords.shape}', flush=True)
        print(f'Pcoords[0]: {pcoords[0]}', flush=True)

        # Make a new model directory for this iteration
        model_dir = output_dir / f'model_{idx:06d}'

        # Fit the model
        print(f'Fitting model at iteration: {idx}', flush=True)
        checkpoint_path = model.fit(
            x=cmaps,
            model_dir=model_dir,
            scalars={'pcoord': pcoords},
        )
        print(f'Finished fitting model at iteration: {idx}', flush=True)
        print(f'Checkpoint path: {checkpoint_path}', flush=True)

        # Construct the train result
        result = TrainResult(
            config_path=config.config_path,
            checkpoint_path=checkpoint_path,
        )

        # Send the new model weights to the thinker
        print(
            f'Sending new model weights to thinker at iteration: {idx}',
            flush=True,
        )
        stream_producer.send(topic=TRAIN_TOPIC, obj=result)

    # NOTE: This final return is not necessary, but it is included
    #       to keep the function signature consistent with the non-streaming.
    return result
