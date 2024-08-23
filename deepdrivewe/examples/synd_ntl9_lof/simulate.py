"""Run a synD simulation and return the pcoord and coordinates."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mdtraj
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix

from deepdrivewe import SimMetadata
from deepdrivewe import SimResult
from deepdrivewe.simulation.synd import SynDConfig
from deepdrivewe.simulation.synd import SynDSimulation
from deepdrivewe.simulation.synd import SynDTrajAnalyzer
from deepdrivewe.stream import ProxyStreamConfig


class SimulationConfig(SynDConfig):
    """Arguments for the SynD contact map analyzer."""

    reference_file: Path = Field(
        description='The reference PDB file for the contact map analysis.',
    )
    contact_cutoff: float = Field(
        default=8.0,
        description='The angstrom cutoff distance for defining contacts.',
    )
    mdtraj_selection: str = Field(
        default='name CA',
        description='The mdtraj selection string for the atoms to use.',
    )


class ContactMapAnalyzer(BaseModel, SynDTrajAnalyzer):
    """Analyze SynD simulations using contact maps."""

    reference_file: Path = Field(
        description='The reference PDB file for the contact map analysis.',
    )
    contact_cutoff: float = Field(
        default=8.0,
        description='The angstrom cutoff distance for defining contacts.',
    )
    mdtraj_selection: str = Field(
        default='name CA',
        description='The mdtraj selection string for the atoms to use.',
    )
    convert_coords_to_angstroms: bool = Field(
        default=True,
        description='Whether to convert the coordinates to angstroms.',
    )

    def get_contact_maps(self, sim: SynDSimulation) -> np.ndarray:
        """Compute contact maps from the trajectory.

        Parameters
        ----------
        sim : SynDSimulation
            The simulation to analyze.

        Returns
        -------
        np.ndarray
            The contact maps from the trajectory (n_steps, *)
            where * is a ragged dimension.
        """
        # Get the atomic coordinates from the aligned trajectory
        coords = self.get_coords(sim)

        # Convert from nm to angstroms
        if self.convert_coords_to_angstroms:
            coords *= 10.0

        # Load the reference structure
        ref_traj: mdtraj.Topology = mdtraj.load_topology(self.reference_file)

        # Get the CA atom indices
        ca_indices = ref_traj.select(self.mdtraj_selection)

        # Index into the CA atom coords (n_steps, n_ca_atoms, 3)
        ca_coords = coords[:, ca_indices]

        # Compute a distance matrix for each frame
        distance_matrices = [distance_matrix(x, x) for x in ca_coords]

        # Convert the distance matrices to contact maps (binary matrices 0-1s)
        contact_maps = np.array(distance_matrices) < self.contact_cutoff

        # Convert the contact maps to sparse matrices for memory efficiency
        coo_contact_maps = [coo_matrix(x) for x in contact_maps]

        # Collect the rows and cols of the sparse matrices
        # (since the values are all 1s, we don't need them)
        rows = [x.row.astype('int16') for x in coo_contact_maps]
        cols = [x.col.astype('int16') for x in coo_contact_maps]

        # Concatenate the row and col indices into a single array
        contact_maps = [np.concatenate(x) for x in zip(rows, cols)]

        # Return the contact maps as a ragged numpy array
        return np.array(contact_maps, dtype=object)


def run_simulation(
    metadata: SimMetadata,
    config: SimulationConfig,
    output_dir: Path,
    stream_config: StreamConfig | None = None,
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

    if stream_config is not None:
        producer = stream_config.get_producer()

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
    analyzer = ContactMapAnalyzer(
        reference_file=config.reference_file,
        contact_cutoff=config.contact_cutoff,
        mdtraj_selection=config.mdtraj_selection,
    )
    pcoords = analyzer.get_pcoords(sim)
    contact_maps = analyzer.get_contact_maps(sim)

    # Update the simulation metadata
    metadata.restart_file = sim.restart_file
    metadata.pcoord = pcoords.tolist()
    metadata.mark_simulation_end()

    # Create the simulation result
    result = SimResult(
        data={'contact_maps': contact_maps, 'pcoords': pcoords},
        metadata=metadata,
    )

    if stream_config:
        producer.send(stream_config.store_name, result, evict=True)

    return result
