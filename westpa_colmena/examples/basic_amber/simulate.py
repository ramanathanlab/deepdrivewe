"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import mdtraj as md
import numpy as np
from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.ensemble import SimulationMetadata
from westpa_colmena.simulation.amber import AmberConfig
from westpa_colmena.simulation.amber import AmberSimulation


class SimulationArgs(BaseModel):
    """Arguments for the naive resampler."""

    amber_config: AmberConfig = Field(
        metadata={
            'help': 'The configuration for the Amber simulation.',
        },
    )
    cpp_traj_exe: Path = Field(
        metadata={
            'help': 'The path to the cpptraj executable.',
        },
    )
    reference_pdb_file: Path = Field(
        metadata={
            'help': 'The reference PDB file for the cpptraj analysis.',
        },
    )


@dataclass
class SimulationResult:
    """Store the results of a single Amber simulation."""

    pcoord: np.ndarray = field(
        metadata={
            'help': 'The progress coordinate for the Amber simulation.',
        },
    )
    coords: np.ndarray = field(
        metadata={
            'help': 'The atomic coordinates for the Amber simulation.',
        },
    )
    metadata: SimulationMetadata = field(
        metadata={
            'help': 'The metadata for the Amber simulation.',
        },
    )


# TODO: Adapt this for the nacl example (we may be able to make a
# generic version of this)
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
            input_file = f'parm {sim.top_file} \n'
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
        traj = md.load(sim.trajectory_file, top=sim.top_file)

        # Load the reference structure
        ref_traj = md.load(self.reference_pdb_file)

        # Align the trajectory to the reference structure
        traj_aligned = traj.superpose(ref_traj)

        # Get the atomic coordinates from the aligned trajectory
        aligned_coordinates = traj_aligned.xyz

        return aligned_coordinates


def run_simulation(
    args: SimulationArgs,
    output_dir: Path,
    metadata: SimulationMetadata,
) -> SimulationResult:
    """Run a simulation and return the pcoord and coordinates."""
    from westpa_colmena.simulation.amber import AmberSimulation

    # Create the simulation output directory
    sim_output_dir = (
        output_dir
        / f'{metadata.iteration_id:06d}'
        / f'{metadata.simulation_id:06d}'
    )
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    # First run the simulation
    simulation = AmberSimulation(
        amber_exe=args.amber_config.amber_exe,
        md_input_file=args.amber_config.md_input_file,
        top_file=args.amber_config.top_file,
        output_dir=sim_output_dir,
        checkpoint_file=metadata.parent_restart_file,
    )

    # Run the simulation
    simulation.run()

    # Then run cpptraj to get the pcoord and coordinates
    analyzer = CppTrajAnalyzer(
        cpp_traj_exe=args.cpp_traj_exe,
        reference_pdb_file=args.reference_pdb_file,
    )
    pcoord = analyzer.get_pcoords(simulation)
    coords = analyzer.get_coords(simulation)

    # Update the simulation metadata
    metadata = metadata.copy()
    metadata.restart_file = simulation.restart_file

    result = SimulationResult(
        pcoord=pcoord,
        coords=coords,
        metadata=metadata,
    )

    return result
