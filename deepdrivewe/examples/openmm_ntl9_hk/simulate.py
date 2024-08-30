"""Simulate a system using Amber and analyze the results using cpptraj."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import Field

from deepdrivewe import BaseModel
from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe.simulation.openmm import ContactMapRMSDReporter
from deepdrivewe.simulation.openmm import OpenMMConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation


class SimulationConfig(BaseModel):
    """Arguments for the naive resampler."""

    openmm_config: OpenMMConfig = Field(
        description='The configuration for the Amber simulation.',
    )
    top_file: Path | None = Field(
        default=None,
        description='The topology file for the simulation.',
    )
    reference_file: Path = Field(
        description='The reference PDB file for the analysis.',
    )
    cutoff_angstrom: float = Field(
        default=8.0,
        description='The angstrom cutoff distance for defining contacts.',
    )
    mda_selection: str = Field(
        default='protein and name CA',
        description='The MDAnalysis selection string for the atoms to use.',
    )
    openmm_selection: Sequence[str] = Field(
        default=('CA',),
        description='The OpenMM selection strings for the atoms to use.',
    )


def run_simulation(
    metadata: SimMetadata,
    config: SimulationConfig,
    output_dir: Path,
) -> SimResult:
    """Run a simulation and return the pcoord and coordinates."""
    # Add performance logging
    metadata.mark_simulation_start()

    print(config, flush=True)

    # Create the simulation output directory
    sim_output_dir = output_dir / metadata.simulation_name

    # Remove the directory if it already exists
    # (this would be from a task failure)
    if sim_output_dir.exists():
        # Remove each file in the directory
        for file in sim_output_dir.iterdir():
            file.unlink()

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

    # Add the contact map and RMSD reporter
    reporter = ContactMapRMSDReporter(
        report_interval=config.openmm_config.report_steps,
        reference_file=config.reference_file,
        cutoff_angstrom=config.cutoff_angstrom,
        mda_selection=config.mda_selection,
        openmm_selection=config.openmm_selection,
    )

    # Run the simulation
    simulation.run(reporters=[reporter])

    # Run the contact map and RMSD analysis
    contact_maps = reporter.get_contact_maps()
    pcoord = reporter.get_rmsds()

    # contact_maps, pcoord = config.analyzer.get_contact_map_and_rmsd(
    # simulation)

    # Update the simulation metadata
    metadata.restart_file = simulation.restart_file
    metadata.pcoord = pcoord.tolist()
    metadata.mark_simulation_end()

    result = SimResult(
        data={'contact_maps': contact_maps, 'pcoord': pcoord},
        metadata=metadata,
    )

    return result
