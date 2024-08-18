"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

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
    sim_output_dir = (
        output_dir
        / f'{metadata.iteration_id:06d}'
        / f'{metadata.simulation_id:06d}'
    )

    # Remove the directory if it already exists
    # (this would be from a task failure)
    if sim_output_dir.exists():
        # Wait a bit to make sure the directory is not being
        # used and avoid .nfs file race conditions
        time.sleep(10)
        shutil.rmtree(sim_output_dir)

    # Create a fresh output directory
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy input files to the output directory
    restart_file = metadata.parent_restart_file
    checkpoint_file = sim_output_dir / f'parent{restart_file.suffix}'
    checkpoint_file = shutil.copy(restart_file, checkpoint_file)
    md_input = shutil.copy(config.amber_config.md_input_file, sim_output_dir)
    top_file = shutil.copy(config.amber_config.top_file, sim_output_dir)

    # Log the yaml config file to this directory
    config.dump_yaml(sim_output_dir / 'config.yaml')

    # Initialize the simulation
    simulation = AmberSimulation(
        amber_exe=config.amber_config.amber_exe,
        md_input_file=md_input,
        top_file=top_file,
        output_dir=sim_output_dir,
        checkpoint_file=checkpoint_file,
    )

    # Run the simulation
    simulation.run()

    # Then run cpptraj to get the pcoord and coordinates
    analyzer = BackboneRMSDAnalyzer(reference_file=config.reference_file)
    pcoord = analyzer.get_pcoords(simulation)
    coords = analyzer.get_coords(simulation)

    # Update the simulation metadata
    metadata.restart_file = simulation.restart_file
    metadata.pcoord = pcoord.tolist()
    # Save the full pcoord data to the auxdata
    metadata.auxdata = {'pcoord': pcoord.tolist()}

    # Log the performance
    metadata.mark_simulation_end()

    result = SimResult(
        pcoord=pcoord,
        coords=coords,
        metadata=metadata,
    )

    return result
