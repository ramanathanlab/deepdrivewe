"""Checkpointing for the weighted ensemble."""

from __future__ import annotations

import json
from pathlib import Path

from westpa_colmena.ensemble import WeightedEnsemble
from westpa_colmena.io import WestpaH5File


class EnsembleCheckpointer:
    """Checkpointer for the weighted ensemble."""

    def __init__(self, output_dir: Path) -> None:
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.h5file = WestpaH5File(westpa_h5file_path=output_dir / 'west.h5')

        # Make the checkpoint directory if it does not exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, weighted_ensemble: WeightedEnsemble) -> None:
        """Save the weighted ensemble to a checkpoint file.

        Parameters
        ----------
        weighted_ensemble : WeightedEnsemble
            The weighted ensemble to save to the checkpoint file.
        """
        # Save the weighted ensemble to a checkpoint file
        iteration = weighted_ensemble.metadata.iteration_id
        filename = f'checkpoint-{iteration:06d}.json'

        # Save the weighted ensemble to the checkpoint file
        with open(self.checkpoint_dir / filename, 'w') as fp:
            fp.write(weighted_ensemble.json(indent=2))

        # Save the weighted ensemble to the HDF5 file
        self.h5file.append(
            cur_sims=weighted_ensemble.cur_sims,
            basis_states=weighted_ensemble.basis_states,
            target_states=weighted_ensemble.target_states,
            metadata=weighted_ensemble.metadata,
        )

    def load(self, path: str | Path | None = None) -> WeightedEnsemble:
        """Load the weighted ensemble from a checkpoint file.

        Returns
        -------
        WeightedEnsemble
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
            return WeightedEnsemble(**json.load(fp))

    def latest_checkpoint(self) -> Path | None:
        """Return the latest checkpoint file.

        Returns
        -------
        Path or None
            The latest checkpoint file or None if no checkpoint file exists.
        """
        return max(self.checkpoint_dir.glob('checkpoint-*.json'), default=None)
