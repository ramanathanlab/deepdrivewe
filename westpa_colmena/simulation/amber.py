"""Run Amber simulations and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import mdtraj as md
import numpy as np
from pydantic import BaseModel
from pydantic import Field


@dataclass
class AmberSimulation:
    """Run an Amber simulation."""

    amber_exe: str = field(
        metadata={'help': 'The path to the Amber executable.'},
    )
    md_input_file: Path = field(
        metadata={'help': 'The input file for the Amber simulation.'},
    )
    top_file: Path = field(
        metadata={'help': 'The prmtop file for the Amber simulation.'},
    )

    # These properties are different for each simulation
    output_dir: Path = field(
        metadata={'help': 'The output directory for the Amber simulation.'},
    )
    checkpoint_file: Path = field(
        metadata={'help': 'The checkpoint file for the Amber simulation.'},
    )

    seed: int | None = field(
        default=None,
        metadata={'help': 'The random seed.'},
    )

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

        # Set the random seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**16)

        # Populate the md_input_file with the random seed
        command = f"sed -i 's/RAND/{self.seed}/g' {self.md_input_file}"
        subprocess.run(command, check=False, shell=True)

        # Setup the simulation
        command = (
            f'{self.amber_exe} -O '
            f'-i {self.md_input_file} '
            f'-o {self.log_file} '
            f'-p {self.top_file} '
            f'-c {self.checkpoint_file} '
            f'-r {self.restart_file} '
            f'-x {self.trajectory_file} '
            f'-inf {self.info_file}'
        )

        # Log the command
        print(command)

        # Run the simulation
        subprocess.run(command, shell=True, check=True, cwd=self.output_dir)


class AmberConfig(BaseModel):
    """Config for an Amber simulation."""

    amber_exe: str = Field(
        default='sander',
        description='The path to the Amber executable.',
    )
    md_input_file: Path = Field(
        description='The input file for the Amber simulation.',
    )
    top_file: Path = Field(
        description='The prmtop file for the Amber simulation.',
    )


def run_cpptraj(command: str) -> list[float]:
    """Run cpptraj with the command and return the progress coordinate.

    Parameters
    ----------
    command : str
        The cpptraj command instructions to run (these get written to a
        cpptraj input file).

    Returns
    -------
    list[float]
        The progress coordinate from the cpptraj output.
    """
    # Make a temporary directory to store the cpptraj inputs and outputs
    with tempfile.TemporaryDirectory() as tmp:
        # Create the cpptraj output file
        output_file = Path(tmp) / 'cpptraj.dat'
        # Format the cpptraj input file contents
        command.format(output_file=output_file)

        # Write the cpptraj input file to a temporary file
        input_file = Path(tmp) / 'cpptraj.in'
        input_file.write_text(command)

        # Run cpptraj
        _command = f'cat {input_file} | cpptraj'
        subprocess.run(_command, shell=True, check=True)

        # Parse the cpptraj output file (first line is a header)
        lines = Path(output_file).read_text().splitlines()[1:]
        # The second column is the progress coordinate
        pcoord = [float(line.split()[1]) for line in lines if line]

    return pcoord


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
        print(
            f'Analyzing simulation traj file {sim.trajectory_file} and'
            f' top file {sim.top_file}',
        )
        # Load the trajectory using mdtraj
        traj = md.load(sim.trajectory_file, top=sim.top_file)

        # Load the reference structure
        ref_traj = md.load(self.reference_file, top=sim.top_file)

        # Align the trajectory to the reference structure
        traj_aligned = traj.superpose(ref_traj)

        # Get the atomic coordinates from the aligned trajectory
        aligned_coordinates = traj_aligned.xyz

        return aligned_coordinates
