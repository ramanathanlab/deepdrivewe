"""Checkpointing for the weighted ensemble."""

from __future__ import annotations

import json
from pathlib import Path

from westpa_colmena.ensemble import WeightedEnsembleV2
from westpa_colmena.io import WestpaH5File


class EnsembleCheckpointer:
    """Checkpointer for the weighted ensemble."""

    def __init__(self, output_dir: Path) -> None:
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.h5file = WestpaH5File(westpa_h5file_path=output_dir / 'west.h5')

        # Make the checkpoint directory if it does not exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, weighted_ensemble: WeightedEnsembleV2) -> None:
        """Save the weighted ensemble to a checkpoint file.

        Parameters
        ----------
        weighted_ensemble : WeightedEnsembleV2
            The weighted ensemble to save to the checkpoint file.
        """
        # Save the weighted ensemble to a checkpoint file
        name = f'checkpoint-{len(weighted_ensemble.simulations):06d}.json'
        checkpoint_file = self.checkpoint_dir / name

        # Save the weighted ensemble to the checkpoint file
        with open(checkpoint_file, 'w') as fp:
            fp.write(weighted_ensemble.json(indent=2))
            # fp.write(weighted_ensemble.json(exclude={'binner_pickle'}))

        # Save the weighted ensemble to the HDF5 file
        self.h5file.append(
            cur_sims=weighted_ensemble.cur_sims,
            basis_states=weighted_ensemble.basis_states,
            target_states=weighted_ensemble.target_states,
            metadata=weighted_ensemble.metadata,
        )

    def load(self, path: Path | None = None) -> WeightedEnsembleV2:
        """Load the weighted ensemble from a checkpoint file.

        Returns
        -------
        WeightedEnsembleV2
            The weighted ensemble loaded from the checkpoint file.

        Raises
        ------
        FileNotFoundError
            If no checkpoint file is found.
        """
        # TODO: In order to resume from a checkpoint in a different
        #       output directory, we need to fix the output_dir
        #       path prefix in each of the SimMetadata, etc objects.

        # Get the latest checkpoint file
        if path is None:
            path = self.latest_checkpoint()
            if path is None:
                raise FileNotFoundError('No checkpoint file found.')

        # Load the weighted ensemble from the checkpoint file
        with open(path) as fp:
            return WeightedEnsembleV2(**json.load(fp))

    def latest_checkpoint(self) -> Path | None:
        """Return the latest checkpoint file.

        Returns
        -------
        Path or None
            The latest checkpoint file or None if no checkpoint file exists.
        """
        return max(self.checkpoint_dir.glob('checkpoint-*.json'), default=None)
