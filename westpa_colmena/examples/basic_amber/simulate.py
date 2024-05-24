"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from westpa_colmena.ensemble import SimulationMetadata
from westpa_colmena.simulation.amber import AmberConfig
from westpa_colmena.simulation.amber import AmberSimulation
from westpa_colmena.simulation.amber import AmberTrajAnalyzer


class SimulationConfig(BaseModel):
    """Arguments for the naive resampler."""

    amber_config: AmberConfig = Field(
        metadata={
            'help': 'The configuration for the Amber simulation.',
        },
    )
    reference_file: Path = Field(
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
        # Make a temporary file to store the cpptraj outputs
        with tempfile.NamedTemporaryFile() as tmp:
            align_file = tmp.name

            # Create the cpptraj input file
            input_file = (
                f'parm {sim.top_file} \n'
                f'trajin {sim.checkpoint_file}\n'
                f'trajin {sim.trajectory_file}\n'
                f'reference {self.reference_file} [reference] \n'
                f'rms ALIGN @CA,C,O,N,H reference out {align_file} \n'
                'go'
            )

            # Run cpptraj
            command = f'echo -e {input_file} | cpptraj'
            subprocess.run(command, shell=True, check=True)

            # Parse the cpptraj output file (first line is a header)
            lines = Path(align_file).read_text().splitlines()[1:]
            # The second column is the progress coordinate
            pcoord = [float(line.split()[1]) for line in lines if line]

        return np.array(pcoord)


class DistanceAnalyzer(AmberTrajAnalyzer):
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
        # Make a temporary file to store the cpptraj outputs
        with tempfile.NamedTemporaryFile() as tmp:
            align_file = tmp.name

            # Create the cpptraj input file
            input_file = (
                f'parm {sim.top_file} \n'
                f'trajin {sim.checkpoint_file}\n'
                f'trajin {sim.trajectory_file}\n'
                f'reference {self.reference_file} [reference] \n'
                f'distance na-cl :1@Na+ :2@Cl- out {align_file} \n'
                'go'
            )
            # Run cpptraj
            command = f'echo -e {input_file} | cpptraj'
            subprocess.run(command, shell=True, check=True)

            # Parse the cpptraj output file (first line is a header)
            lines = Path(align_file).read_text().splitlines()[1:]
            # The second column is the progress coordinate
            pcoord = [float(line.split()[1]) for line in lines if line]

        return np.array(pcoord)


def run_simulation(
    metadata: SimulationMetadata,
    config: SimulationConfig,
    output_dir: Path,
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

    # TODO: Copy input file and checkpoint file to this directory.
    # TODO: Log the yaml config file to this directory

    # First run the simulation
    simulation = AmberSimulation(
        amber_exe=config.amber_config.amber_exe,
        md_input_file=config.amber_config.md_input_file,
        top_file=config.amber_config.top_file,
        output_dir=sim_output_dir,
        checkpoint_file=metadata.parent_restart_file,
    )

    # Run the simulation
    simulation.run()

    # Then run cpptraj to get the pcoord and coordinates
    analyzer = DistanceAnalyzer(reference_file=config.reference_file)
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
