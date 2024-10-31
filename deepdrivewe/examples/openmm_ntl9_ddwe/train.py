"""Training module."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from deepdrivewe import SimResult
from deepdrivewe import TrainResult
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
            sim_output=sim_output,
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

    # Extract the last frame contact maps and rmsd from each simulation
    contact_maps = np.concatenate(
        [sim.data['contact_maps'] for sim in sim_output],
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
    sim_output: list[SimResult],
    config: TrainConfig,
    output_dir: Path,
    stream_config: ProxyStreamConfig,
) -> TrainResult:
    """Train the model on the simulation output."""
    # Make the output directory
    itetation = sim_output[0].metadata.iteration_id
    output_dir = output_dir / f'{itetation:06d}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream consumer for getting new simulation data
    stream_consumer = stream_config.get_consumer(topic=SIMULATION_TOPIC)
    # Stream producer for sending new trained model weights to the thinker
    stream_producer = stream_config.get_producer(topic=TRAIN_TOPIC)

    # TODO: Decide how much data we want to keep in the re-train history.
    contact_map_history = []
    pcoord_history = []

    # Loop indefinitely until we get a stop iteration from the stream consumer
    for idx in itertools.count():
        # If we have reached the retrain interval, re-initialize the trainer
        # NOTE: This always happens on the first iteration
        if idx % config.stream_retrain_interval == 0:
            # Load the model configuration
            model_config = ConvolutionalVAEConfig.from_yaml(config.config_path)

            # Load the model
            model = ConvolutionalVAE(
                model_config,
                checkpoint_path=config.checkpoint_path,
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
            break

        # Extract the contact maps and rmsd from each simulation
        contact_maps = np.concatenate([x['contact_maps'] for x in items])
        pcoords = np.concatenate([x['pcoords'] for x in items])
        pcoords = pcoords.flatten()

        # TODO: It might be necessary to put these into a numpy array
        # Concatenate the new data with the history
        contact_map_history.extend(contact_maps)
        pcoord_history.extend(pcoords)

        # Fit the model
        checkpoint_path = model.fit(
            x=contact_map_history,
            model_dir=output_dir / 'model',
            scalars={'pcoord': pcoord_history},
        )

        # Construct the train result
        result = TrainResult(
            config_path=config.config_path,
            checkpoint_path=checkpoint_path,
        )

        # Send the new model weights to the thinker
        stream_producer.send(topic=TRAIN_TOPIC, obj=result)

    # NOTE: This final return is not necessary, but it is included
    #       to keep the function signature consistent with the non-streaming.
    return result
