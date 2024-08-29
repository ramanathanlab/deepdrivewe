"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe.simulation.openmm import ContactMapRMSDAnalyzer
from deepdrivewe.simulation.openmm import OpenMMConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation


class SimulationConfig(BaseModel):
    """Arguments for the naive resampler."""

    openmm_config: OpenMMConfig = Field(
        description='The configuration for the Amber simulation.',
    )
    analyzer: ContactMapRMSDAnalyzer = Field(
        description='The configuration for the contact map and RMSD analysis.',
    )
    top_file: Path | None = Field(
        default=None,
        description='The topology file for the simulation.',
    )


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
    simulation = OpenMMSimulation(
        config=config.openmm_config,
        top_file=config.top_file,
        output_dir=sim_output_dir,
        checkpoint_file=metadata.parent_restart_file,
    )

    # Run the simulation
    simulation.run()

    # Run the contact map and RMSD analysis
    contact_maps, pcoord = config.analyzer.get_contact_map_and_rmsd(simulation)

    # Update the simulation metadata
    metadata.restart_file = simulation.restart_file
    metadata.pcoord = pcoord.tolist()
    metadata.mark_simulation_end()

    result = SimResult(
        data={'contact_maps': contact_maps, 'pcoord': pcoord},
        metadata=metadata,
    )

    return result
