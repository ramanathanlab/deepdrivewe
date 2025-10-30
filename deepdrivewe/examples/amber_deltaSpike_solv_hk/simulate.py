"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mdtraj
import numpy as np
from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe.simulation.amber import AmberConfig
from deepdrivewe.simulation.amber import AmberSimulation
from deepdrivewe.simulation.amber import AmberTrajAnalyzer
from deepdrivewe.simulation.amber import run_cpptraj


class SimulationConfig(BaseModel):
    """Arguments for the naive resampler."""

    amber_config: AmberConfig = Field(
        description='The configuration for the Amber simulation.',
    )
    reference_file: Path = Field(
        description='The reference PDB file for the cpptraj analysis.',
    )
    analysis_file: Path = Field(
        description='The input file for the cpptraj analysis.',
    )


class RBDRMSDCOMAnalyzer(AmberTrajAnalyzer):
    """Analyze Amber simulations using cpptraj."""

    analysis_file: Path = Field(
        description='The input file for the cpptraj analysis.',
    )
    reference_file: Path = Field(
        description='reference file for analysis',
    )

    def run_cpptraj(self, command: str, verbose: bool = False) -> list[float]:
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
            #command = command.format(output_file=output_file)

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
                    cwd=Path(tmp),
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
            rmsd = np.loadtxt(Path(tmp) / 'rbd_rmsdA.dat', skiprows=1, usecols=1)
            com = np.loadtxt(Path(tmp) / 'rbd_comA.dat', skiprows=1, usecols=1)
            pcoord = np.dstack((com, rmsd))[0]

        return pcoord

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
        # Create the cpptraj command file
        with open(self.analysis_file, 'r') as f:
            command = f.read()

        load_command = (
            f'parm {sim.top_file}\n'
            f'trajin {sim.output_dir}/parent.ncrst\n'
            f'trajin {sim.trajectory_file}\n'
            f'autoimage\n\n'
            f'parm {self.reference_file} [open]\n'
            f'reference {self.reference_file} parm {self.reference_file} [open]\n'
        )

        command = load_command + command
        pcoord = self.run_cpptraj(command)

        return pcoord

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
        traj = mdtraj.load(sim.trajectory_file, top=sim.top_file)
        #traj_aligned = mdtraj.load(sim.trajectory_file, top=sim.top_file)

        # Load the reference structure
        ref_traj = mdtraj.load(self.reference_file, top=self.reference_file)

        # Align the trajectory to the reference structure
        traj_aligned = traj.superpose(ref_traj, atom_indices=ref_traj.top.select('protein and name CA'))

        # Get the atomic coordinates from the aligned trajectory
        aligned_coordinates = traj_aligned.xyz

        return aligned_coordinates



class BackboneRMSDAnalyzer(AmberTrajAnalyzer):
    """Analyze Amber simulations using cpptraj."""

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
        # Create the cpptraj command file
        command = (
            f'parm {sim.top_file}\n'
            f'trajin {sim.trajectory_file}\n'
            f'reference {self.reference_file} [reference]\n'
            'rms @CA reference out {output_file}\n'
            'go'
        )

        # Run the command
        pcoords = run_cpptraj(command)

        return np.array(pcoords).reshape(-1, 1)


def run_simulation(
    metadata: SimMetadata,
    config: SimulationConfig,
    output_dir: Path,
) -> SimResult:
    """Run a simulation and return the pcoord and coordinates."""
    # Add performance logging
    metadata.mark_simulation_start()

    # Create the simulation output directory
    sim_output_dir = output_dir / metadata.simulation_name

    # Remove the directory if it already exists
    # (this would be from a task failure)
    if sim_output_dir.exists():
        # Wait a bit to make sure the directory is not being
        # used and avoid .nfs file race conditions
        time.sleep(10)
        shutil.rmtree(sim_output_dir)

    # Create a fresh output directory
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    # Log the yaml config file to this directory
    config.dump_yaml(sim_output_dir / 'config.yaml')

    # Initialize the simulation
    simulation = AmberSimulation(
        amber_exe=config.amber_config.amber_exe,
        input_file=config.amber_config.input_file,
        top_file=config.amber_config.top_file,
        output_dir=sim_output_dir,
        checkpoint_file=metadata.parent_restart_file,
    )

    # Run the simulation
    simulation.run()

    # Then run cpptraj to get the pcoord and coordinates
    analyzer = RBDRMSDCOMAnalyzer(reference_file=config.reference_file, analysis_file=config.analysis_file)
    pcoord = analyzer.get_pcoords(simulation)
    coords = analyzer.get_coords(simulation)

    # Update the simulation metadata
    metadata.restart_file = simulation.restart_file
    metadata.pcoord = pcoord.tolist()
    metadata.mark_simulation_end()

    metadata.model_validate(obj=metadata)

    result = SimResult(
        data={'coords': coords, 'pcoord': pcoord},
        metadata=metadata,
    )

    return result
