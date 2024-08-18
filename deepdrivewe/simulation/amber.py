"""Run Amber simulations and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np
from pydantic import BaseModel
from pydantic import Field


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


class AmberSimulation(BaseModel):
    """Run an Amber simulation."""

    amber_exe: str = Field(
        description='The path to the Amber executable.',
    )
    md_input_file: Path = Field(
        description='The input file for the Amber simulation.',
    )
    top_file: Path = Field(
        description='The prmtop file for the Amber simulation.',
    )

    # These properties are different for each simulation
    output_dir: Path = Field(
        description='The output directory for the Amber simulation.',
    )
    checkpoint_file: Path = Field(
        description='The checkpoint file for the Amber simulation.',
    )
    seed: int | None = Field(
        default=None,
        description='The random seed.',
    )

    @property
    def trajectory_file(self) -> Path:
        """The trajectory file for the Amber simulation."""
        return self.output_dir / 'seg.nc'

    @property
    def restart_file(self) -> Path:
        """The restart file for the Amber simulation."""
        return self.output_dir / f'seg{self.checkpoint_file.suffix}'

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

        # Create stderr log file (by default, stdout is captured
        # in the log file).
        stderr = self.output_dir / 'stderr.log'

        # Set the random seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2**16)

        # Populate the md_input_file with the random seed
        command = f"sed -i 's/RAND/{self.seed}/g' {self.md_input_file}"
        with open(stderr, 'a') as err:
            subprocess.run(
                command,
                check=False,
                shell=True,
                stdout=err,
                stderr=err,
            )

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
        with open(stderr, 'a') as err:
            subprocess.run(
                command,
                shell=True,
                check=True,
                cwd=self.output_dir,
                stdout=err,
                stderr=err,
            )


def run_cpptraj(command: str, verbose: bool = False) -> list[float]:
    """Run cpptraj with the command and return the progress coordinate.

    Parameters
    ----------
    command : str
        The cpptraj command instructions to run (these get written to a
        cpptraj input file).
    verbose : bool
        Whether to print the stdout and stderr of the cpptraj command
        (by default False).

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
        command = command.format(output_file=output_file)

        # Write the cpptraj input file to a temporary file
        input_file = Path(tmp) / 'cpptraj.in'
        input_file.write_text(command)

        # Capture the stdout and stderr
        stdout = Path(tmp) / 'stdout.log'
        stderr = Path(tmp) / 'stderr.log'

        # Run cpptraj
        _command = f'cat {input_file} | cpptraj'

        # Run the command and capture the output
        with open(stdout, 'a') as out, open(stderr, 'a') as err:
            result = subprocess.run(
                _command,
                shell=True,
                # Do not raise an exception on a non-zero return code
                check=False,
                stdout=out,
                stderr=err,
            )

        # Check the return code
        if result.returncode != 0:
            print(
                f'Command: {_command}\nfailed '
                f'with return code {result.returncode}.',
            )

        # Print the stdout and stderr
        if verbose or result.returncode != 0:
            with open(stdout) as out, open(stderr) as err:
                print(f'{out.read()}\n\n{err.read()}')

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
