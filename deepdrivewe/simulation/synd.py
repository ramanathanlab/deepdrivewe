"""Run a SynD simulation.

For more information, see the SynD documentation at:
https://github.com/jdrusso/SynD/tree/main
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe.workflows.registry import register


class SynDConfig(BaseModel):
    """Arguments for the naive resampler."""

    synd_model_file: Path = Field(
        description='The path to the SynD model (.pkl/.synd) file.',
    )
    n_steps: int = Field(
        default=2,
        ge=2,
        description='The number of steps to run the simulation.',
    )


@register()  # type: ignore[arg-type]
class SynDSimulation:
    """Run a SynD simulation."""

    def __init__(self, synd_model_file: Path, n_steps: int):
        """Initialize the SynD simulation.

        Parameters
        ----------
        synd_model_file : Path
            The path to the SynD model (pkl) file.
        n_steps : int
            The number of steps to run the simulation.
        """
        self.synd_model_file = synd_model_file
        self.n_steps = n_steps

        from synd.core import load_model
        from synd.models.discrete.markov import MarkovGenerator

        # Load the SynD model
        self.model: MarkovGenerator = load_model(str(self.synd_model_file))

        # The trajectory for the simulation
        self._traj: np.ndarray | None = None
        self._output_dir: Path | None = None

    @property
    def traj(self) -> np.ndarray:
        """The trajectory for the simulation."""
        if self._traj is None:
            raise ValueError(
                'The trajectory has not been set. Call `run` first.',
            )
        return self._traj

    @property
    def output_dir(self) -> Path:
        """The output directory for the simulation."""
        if self._output_dir is None:
            raise ValueError(
                'The output directory has not been set. Call `run` first.',
            )
        return self._output_dir

    @property
    def trajectory_file(self) -> Path:
        """The trajectory file for the simulation."""
        return self.output_dir / 'seg.npy'

    @property
    def restart_file(self) -> Path:
        """The restart file for the simulation."""
        return self.output_dir / 'checkpoint.npy'

    @property
    def parent_file(self) -> Path:
        """The checkpoint file for the Amber simulation."""
        return self.output_dir / 'parent.npy'

    def run(self, checkpoint_file: Path, output_dir: Path) -> None:
        """Run a SynD simulation."""
        # Set the output directory
        self._output_dir = output_dir

        # Create the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy the checkpoint file to the output directory
        shutil.copy(checkpoint_file, self.parent_file)

        # Load the initial states from the checkpoint file
        # (an array of integers with shape (1,) storing the
        # state index to start from).
        initial_states = np.load(self.parent_file)

        # Generate a trajectory with shape (1, n_steps) storing the
        # state indices for each step (i.e., an integer array with
        # the first element being the initial state index).
        traj = self.model.generate_trajectory(
            initial_states=initial_states,
            n_steps=self.n_steps,
        )

        # NOTE: The first dimension from SynD is the number of trajectories
        # being generated. In our case, we are only generating one trajectory
        # at a time, so we reshape the trajectory to have shape (n_steps, ...).

        # Write the restart file for the next iteration storing the
        # last state from the trajectory as an integer array with shape (1,).
        np.save(self.restart_file, traj[0][-1:])

        # Cache the trajectory analysis
        self._traj = traj


class SynDTrajAnalyzer:
    """Strategy for analyzing SynD trajectories."""

    def get_pcoords(self, sim: SynDSimulation) -> np.ndarray:
        """Get the progress coordinate from the trajectory.

        Parameters
        ----------
        sim : SynDSimulation
            The simulation to analyze.

        Returns
        -------
        np.ndarray
            The progress coordinate from the trajectory (n_steps, pcoord_dim).
        """
        # The progress coordinate with shape (1, n_steps, pcoord_dim)
        pcoords = sim.model.backmap(sim.traj)
        pcoords = pcoords.reshape(sim.n_steps, -1)
        return pcoords

    def get_coords(self, sim: SynDSimulation) -> np.ndarray:
        """Get the atomic coordinates from the aligned trajectory.

        Parameters
        ----------
        sim : SynDSimulation
            The simulation to analyze.

        Returns
        -------
        np.ndarray
            The atomic coordinates from the aligned trajectory
            (n_steps, n_atoms, 3).
        """
        # The full coordinates with shape (1, n_steps, n_atoms, 3)
        coords = sim.model.backmap(sim.traj, mapper='full_coordinates')
        coords = coords.reshape(sim.n_steps, -1, 3)
        return coords


class SynDBasisStateInitializer:
    """SynD basis state initialization."""

    def __init__(self, config: SynDConfig) -> None:
        """Initialize the basis state initializer."""
        # Initialize the simulation to use the backmap function
        self.sim = SynDSimulation(
            synd_model_file=config.synd_model_file,
            n_steps=config.n_steps,
        )

    def __call__(self, basis_file: str) -> list[float]:
        """Initialize the basis state parent coordinates."""
        # Load the basis state with shape ()
        state = np.load(basis_file)

        # Get the pcoord for the basis state
        pcoords = self.sim.model.backmap(state)
        pcoords = pcoords.reshape(-1).tolist()

        return pcoords


def generate_basis_states(state_indices: list[int], output_dir: Path) -> None:
    """Generate the basis state files for the SynD simulation.

    Parameters
    ----------
    state_indices : list[int]
        The state indices to use as basis states.
    output_dir : Path
        The output directory to store the basis states.
    """
    for idx, state_idx in enumerate(state_indices):
        # Create the output directory
        bstate_dir = output_dir / f'bstate-{idx}'
        bstate_dir.mkdir(parents=True, exist_ok=True)

        # Save the basis state as an integer array with shape (1,)
        np.save(bstate_dir / f'bstate-{idx}.npy', np.array([state_idx]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a SynD simulation.')
    parser.add_argument(
        '--basis-states',
        type=Path,
        required=True,
        help='The basis states file (.txt) with one integer index per line.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='The output directory to store the basis states.',
    )

    args = parser.parse_args()

    # Load the basis states from the text file
    text = Path(args.basis_states).read_text()
    state_indices = [int(line) for line in text.splitlines()]

    # Convert the state indices to basis state npy files
    generate_basis_states(state_indices, args.output_dir)
