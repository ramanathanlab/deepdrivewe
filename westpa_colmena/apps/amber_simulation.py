"""Run Amber simulations and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np


@dataclass
class AmberSimulation:
    """Run an Amber simulation."""

    amber_exe: Path
    md_input_file: Path
    prmtop_file: Path

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
        f'-p {self.prmtop_file} '
        f'-c {self.checkpoint_file} '
        f'-r {self.restart_file} '
        f'-x {self.trajectory_file} '
        f'-o {self.log_file} '
        f'-inf {self.info_file}'

        # Run the simulation
        subprocess.run(command, shell=True, check=True)


@dataclass
class CppTrajAnalyzer:
    """Analyze Amber simulations using cpptraj."""

    cpp_traj_exe: Path
    reference_pdb_file: Path

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
        # Make a temporary file to store the cpptraj outputs
        with tempfile.NamedTemporaryFile() as tmp:
            align_file = tmp.name

            # Create the cpptraj input file
            input_file = f'parm {sim.prmtop_file} \n'
            input_file += f'trajin {sim.checkpoint_file}\n'
            input_file += f'trajin {sim.trajectory_file}\n'
            input_file += f'reference {self.reference_pdb_file} [reference] \n'
            input_file += (
                f'rms ALIGN @CA,C,O,N,H reference out {align_file} \n'
            )
            input_file += 'go'

            # Run cpptraj
            command = f'echo -e {input_file} | {self.cpp_traj_exe}'
            subprocess.run(command, shell=True, check=True)

            # Parse the cpptraj output file (first line is a header)
            lines = Path(align_file).read_text().splitlines()[1:]
            # The second column is the progress coordinate
            pcoord = [float(line.split()[1]) for line in lines if line]

        return np.array(pcoord)

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
        traj = md.load(sim.trajectory_file, top=sim.prmtop_file)

        # Load the reference structure
        ref_traj = md.load(self.reference_pdb_file)

        # Align the trajectory to the reference structure
        traj_aligned = traj.superpose(ref_traj)

        # Get the atomic coordinates from the aligned trajectory
        aligned_coordinates = traj_aligned.xyz

        return aligned_coordinates


@dataclass
class SimulationResult:
    """Store the results of a single Amber simulation."""

    pcoord: np.ndarray
    coords: np.ndarray
    restart_file: Path | None
    parent_restart_file: Path
