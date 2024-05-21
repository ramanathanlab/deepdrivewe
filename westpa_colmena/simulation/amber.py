"""Run Amber simulations and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import mdtraj as md
import numpy as np
from pydantic import BaseModel


@dataclass
class AmberSimulation:
    """Run an Amber simulation."""

    amber_exe: str
    md_input_file: Path
    top_file: Path

    # This property is different for each simulation
    output_dir: Path
    checkpoint_file: Path

    @property
    def trajectory_file(self) -> Path:
        """The trajectory file for the Amber simulation."""
        return self.output_dir / 'seg.nc'

    @property
    def restart_file(self) -> Path:
        """The restart file for the Amber simulation."""
        return self.output_dir / 'seg.ncrst'

    @property
    def log_file(self) -> Path:
        """The log file for the Amber simulation."""
        return self.output_dir / 'seg.log'

    @property
    def info_file(self) -> Path:
        """The metadata file for the Amber simulation."""
        return self.output_dir / 'seg.nfo'

    def run(self) -> None:
        """Run an Amber simulation.

        Implementation of the following bash command:
        $PMEMD -O -i md.in -p hmr.prmtop -c parent.ncrst \
            -r seg.ncrst -x seg.nc -o seg.log -inf seg.nfo
        """
        # Create the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup the simulation
        command = f'{self.amber_exe} -O '
        f'-i {self.md_input_file} '
        f'-p {self.top_file} '
        f'-c {self.checkpoint_file} '
        f'-r {self.restart_file} '
        f'-x {self.trajectory_file} '
        f'-o {self.log_file} '
        f'-inf {self.info_file}'

        # Run the simulation
        subprocess.run(command, shell=True, check=True)


class AmberConfig(BaseModel):
    """Config for an Amber simulation."""

    amber_exe: str = field(
        default='sander',
        metadata={
            'help': 'The path to the Amber executable.',
        },
    )
    md_input_file: Path = field(
        metadata={
            'help': 'The input file for the Amber simulation.',
        },
    )
    top_file: Path = field(
        metadata={
            'help': 'The prmtop file for the Amber simulation.',
        },
    )


@dataclass
class AmberTrajAnalyzer(ABC):
    """Strategy for analyzing Amber trajectories."""

    reference_file: Path

    @abstractmethod
    def get_pcoords(self, sim: AmberSimulation) -> np.ndarray:
        """Get the progress coordinate from the aligned trajectory.

        Parameters
        ----------
        sim : AmberSimulation
            The Amber simulation to analyze.

        Returns
        -------
        np.ndarray
            The progress coordinate from the aligned trajectory.
        """
        ...

    def get_coords(self, sim: AmberSimulation) -> np.ndarray:
        """Get the atomic coordinates from the aligned trajectory.

        Parameters
        ----------
        sim : AmberSimulation
            The Amber simulation to analyze.

        Returns
        -------
        np.ndarray
            The atomic coordinates from the aligned trajectory.
        """
        # Load the trajectory using mdtraj
        traj = md.load(sim.trajectory_file, top=sim.top_file)

        # Load the reference structure
        ref_traj = md.load(self.reference_file)

        # Align the trajectory to the reference structure
        traj_aligned = traj.superpose(ref_traj)

        # Get the atomic coordinates from the aligned trajectory
        aligned_coordinates = traj_aligned.xyz

        return aligned_coordinates
