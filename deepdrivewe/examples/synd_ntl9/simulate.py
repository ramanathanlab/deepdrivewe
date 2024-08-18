"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe.simulation.synd import SynDConfig
from deepdrivewe.simulation.synd import SynDSimulation
from deepdrivewe.simulation.synd import SynDTrajAnalyzer


def run_simulation(
    metadata: SimMetadata,
    config: SynDConfig,
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
    sim = SynDSimulation(
        synd_model_file=config.synd_model_file,
        n_steps=config.n_steps,
    )

    # Run the simulation
    sim.run(
        checkpoint_file=metadata.parent_restart_file,
        output_dir=sim_output_dir,
    )

    # Analyze the trajectory
    analyzer = SynDTrajAnalyzer()
    pcoord = analyzer.get_pcoords(sim)
    coords = analyzer.get_coords(sim)

    # Update the simulation metadata
    metadata.restart_file = sim.restart_file
    metadata.pcoord = pcoord.tolist()
    metadata.mark_simulation_end()

    # Return the results
    result = SimResult(
        pcoord=pcoord,
        coords=coords,
        metadata=metadata,
    )

    return result
