"""OpenMM simulation module."""

from __future__ import annotations

import random
import shutil
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Sequence

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

import MDAnalysis
import numpy as np
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import rms
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from deepdrivewe.workflows.utils import retry_on_exception

try:
    import openmm
    import openmm.unit as u
    from openmm import app
except ImportError:
    pass  # For testing purposes


class OpenMMReporter(ABC):
    """Reporter protocol for OpenMM simulations."""

    def __init__(self, report_interval: int) -> None:
        """Initialize the reporter.

        Parameters
        ----------
        report_interval : int
            The interval at which to write frames.
        """
        self.report_interval = report_interval

    def describeNextReport(  # noqa: N802
        self,
        simulation: app.Simulation,
    ) -> tuple[int, bool, bool, bool, bool, bool | None]:
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        step_progress = simulation.currentStep % self.report_interval
        steps = self.report_interval - step_progress
        return (steps, True, False, False, False, None)

    @abstractmethod
    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        pass


class OpenMMConfig(BaseModel):
    """Configuration for an OpenMM simulation."""

    simulation_length_ns: float = Field(
        default=0.01,  # 0.01 ns = 10 ps
        description='The length of the simulation (in nanoseconds).',
    )
    report_interval_ps: float = Field(
        default=2.0,
        description='The report interval for the simulation (in picoseconds).',
    )
    dt_ps: float = Field(
        default=0.002,
        description='The timestep for the simulation.',
    )
    temperature_kelvin: float = Field(
        default=300.0,
        description='The temperature for the simulation.',
    )
    heat_bath_friction_coef: float = Field(
        default=1.0,
        description='The heat bath friction coefficient for the simulation.',
    )
    solvent_type: str = Field(
        default='implicit',
        description='The solvent type for the simulation.',
    )
    explicit_barostat: str | None = Field(
        default=None,
        description='The barostat used for an explicit solvent simulation. '
        'Options are: None (NVT), MonteCarloBarostat, ,'
        'MonteCarloAnisotropicBarostat.',
    )
    run_minimization: bool = Field(
        default=True,
        description='Whether to run energy minimization.',
    )
    set_positions: bool = Field(
        default=True,
        description='Whether to set positions.',
    )
    randomize_velocities: bool = Field(
        default=True,
        description='Whether to randomize the basis state initial velocities.',
    )
    hardware_platform: str = Field(
        default='CUDA',
        description='The hardware platform to use for the simulation.'
        ' Options are: CUDA, OpenCL, CPU.',
    )

    @model_validator(mode='after')
    def validate_explicit_barostat(self) -> Self:
        """Check for valid explicit_barostat options."""
        valid_barostats = (
            None,
            'MonteCarloBarostat',
            'MonteCarloAnisotropicBarostat',
        )
        if (
            self.solvent_type == 'explicit'
            and self.explicit_barostat not in valid_barostats
        ):
            raise ValueError(
                f'Invalid explicit_barostat option: {self.explicit_barostat}',
                f'For explicit solvent, valid options are: {valid_barostats}',
            )
        return self

    @model_validator(mode='after')
    def validate_hardware_platform(self) -> Self:
        """Check for valid hardware_platform options."""
        valid_platforms = ('CUDA', 'OpenCL', 'CPU')
        if self.hardware_platform not in valid_platforms:
            raise ValueError(
                f'Invalid hardware_platform option: {self.hardware_platform}',
                f'Valid options are: {valid_platforms}',
            )
        return self

    @property
    def num_steps(self) -> int:
        """The number of steps to run the simulation."""
        dt_ps = self.dt_ps * u.picoseconds
        simulation_length_ns = self.simulation_length_ns * u.nanoseconds
        return int(simulation_length_ns / dt_ps)

    @property
    def report_steps(self) -> int:
        """The number of steps between log reports."""
        dt_ps = self.dt_ps * u.picoseconds
        report_interval_ps = self.report_interval_ps * u.picoseconds
        return int(report_interval_ps / dt_ps)

    def load_explicit_system_from_top(
        self,
        top_file: str | Path,
    ) -> tuple[openmm.System, app.Topology]:
        """Load an explicit solvent system from a topology file.

        Parameters
        ----------
        top_file : str | Path
            The topology file to load the system from.

        Returns
        -------
        tuple[openmm.System, app.Topology]
            The OpenMM system and topology.
        """
        # Load the topology file
        top = app.AmberPrmtopFile(str(top_file))

        # Configure system
        system = top.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

        return system, top.topology

    def load_implicit_system_from_pdb(
        self,
        pdb_file: str | Path,
    ) -> tuple[openmm.System, app.Topology]:
        """Load an implicit solvent system from a PDB file.

        Parameters
        ----------
        pdb_file : str | Path
            The PDB file to load the system from.

        Returns
        -------
        tuple[openmm.System, app.Topology]
            The OpenMM system and topology.
        """
        # Load the PDB file
        pdb = app.PDBFile(str(pdb_file))

        # Get the topology
        topology = pdb.topology

        # Set the forcefield
        forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')

        # Configure the system
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

        return system, topology

    def load_implicit_system_from_top(
        self,
        top_file: str | Path,
    ) -> tuple[openmm.System, app.Topology]:
        """Load an implicit solvent system from a topology file.

        Parameters
        ----------
        top_file : str | Path
            The topology file to load the system from.

        Returns
        -------
        tuple[openmm.System, app.Topology]
            The OpenMM system and topology.
        """
        # Load the topology file
        top = app.AmberPrmtopFile(str(top_file))

        # Configure the system
        system = top.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
            implicitSolvent=app.OBC1,
        )

        return system, top.topology

    def configure_hardware(self) -> tuple[openmm.Platform, dict[str, str]]:
        """Configure the hardware for the simulation.

        Returns
        -------
        tuple[openmm.Platform, dict[str, str]]
            The OpenMM platform and the platform properties.
        """
        if self.hardware_platform == 'CUDA':
            # Use the CUDA platform
            platform = openmm.Platform.getPlatformByName('CUDA')
            platform_properties = {
                'DeviceIndex': '0',
                'CudaPrecision': 'mixed',
            }
        elif self.hardware_platform == 'OpenCL':
            # Use the OpenCL platform
            platform = openmm.Platform.getPlatformByName('OpenCL')
            platform_properties = {'DeviceIndex': '0'}
        else:
            # Use the CPU platform
            platform = openmm.Platform.getPlatformByName('CPU')
            platform_properties = {}

        return platform, platform_properties

    def configure_integrator(self) -> openmm.LangevinIntegrator:
        """Configure the integrator for the simulation.

        Returns
        -------
        openmm.LangevinIntegrator
            The configured integrator.
        """
        # Configure the integrator
        integrator = openmm.LangevinIntegrator(
            self.temperature_kelvin * u.kelvin,
            self.heat_bath_friction_coef / u.picosecond,
            self.dt_ps * u.picosecond,
        )

        # Set the constraint tolerance
        integrator.setConstraintTolerance(0.00001)

        return integrator

    def configure_barostat(
        self,
    ) -> (
        openmm.MonteCarloBarostat | openmm.MonteCarloAnisotropicBarostat | None
    ):
        """Configure the barostat for the simulation.

        Returns
        -------
        openmm.MonteCarloBarostat | openmm.MonteCarloAnisotropicBarostat | None
            The configured barostat or None if no barostat is used.
        """
        if self.explicit_barostat == 'MonteCarloBarostat':
            return openmm.MonteCarloBarostat(
                1 * u.bar,
                self.temperature_kelvin * u.kelvin,
            )

        elif self.explicit_barostat == 'MonteCarloAnisotropicBarostat':
            return openmm.MonteCarloAnisotropicBarostat(
                (1, 1, 1) * u.bar,
                self.temperature_kelvin * u.kelvin,
                False,
                False,
                True,
            )

        return None

    # NOTE: Add a retry decorator to the method since sometimes the
    # PDB file is not fully written before it is read.
    @retry_on_exception(wait_time=30)
    def configure_simulation(
        self,
        pdb_file: str | Path,
        top_file: str | Path | None,
    ) -> app.Simulation:
        """Configure an OpenMM simulation.

        Parameters
        ----------
        pdb_file : str | Path
            The PDB file to initialize the positions (used to load the system
            topology for implicit solvent).
        top_file : str | Path | None
            The optional topology file to initialize the systems topology
            (required for explicit solvent).

        Returns
        -------
        app.Simulation
            Configured OpenMM Simulation object.

        Raises
        ------
        ValueError
            If explicit solvent is selected and no topology file is provided.
        """
        # Select implicit or explicit solvent configuration and load the system
        if self.solvent_type == 'explicit':
            if top_file is None:
                raise ValueError(
                    'Topology file must be provided for explicit solvent.',
                )
            system, topology = self.load_explicit_system_from_top(top_file)
        elif top_file is not None:
            system, topology = self.load_implicit_system_from_top(top_file)
        else:
            system, topology = self.load_implicit_system_from_pdb(pdb_file)

        # Configure the integrator
        integrator = self.configure_integrator()

        # Configure the barostat
        barostat = self.configure_barostat()
        if barostat is not None:
            system.addForce(barostat)

        # Configure the hardware
        platform, platform_properties = self.configure_hardware()

        # Create the simulation
        sim = app.Simulation(
            topology,
            system,
            integrator,
            platform,
            platform_properties,
        )

        # Set the positions
        if self.set_positions:
            pdb = app.PDBFile(str(pdb_file))
            sim.context.setPositions(pdb.getPositions())

        # Minimize energy and equilibrate
        if self.run_minimization:
            sim.minimizeEnergy()

        # Set velocities to temperature
        if self.randomize_velocities:
            sim.context.setVelocitiesToTemperature(
                self.temperature_kelvin * u.kelvin,
                random.randint(1, 10000),
            )

        return sim


class OpenMMSimulation(BaseModel):
    """OpenMM simulation."""

    config: OpenMMConfig = Field(
        description='The configuration for the OpenMM simulation.',
    )
    checkpoint_file: Path = Field(
        description='The checkpoint file for the simulation.',
    )
    top_file: Path | None = Field(
        default=None,
        description='The topology file for the simulation.',
    )
    output_dir: Path = Field(
        description='The output directory for the simulation.',
    )
    copy_input_files: bool = Field(
        default=True,
        description='Whether to copy the input files to the output directory.'
        'top_file will be copied by default.',
    )

    @property
    def trajectory_file(self) -> Path:
        """The trajectory file for the simulation."""
        return self.output_dir / 'seg.dcd'

    @property
    def restart_file(self) -> Path:
        """The restart file for the simulation.

        NOTE: In the case of OpenMM, this PDB file is used
        to initialize the simulation, and used as a proxy
        for the restart file (which is a checkpoint file).
        The actual checkpoint file is seg.chk (found in the
        same directory as the checkpoint file) which is used
        to save the simulation state and is automatically
        loaded if it exists.
        """
        return self.output_dir / 'seg.pdb'

    @property
    def log_file(self) -> Path:
        """The log file for the simulation."""
        return self.output_dir / 'seg.log'

    def run(self, reporters: list[OpenMMReporter] | None = None) -> None:
        """Run the simulation.

        Parameters
        ----------
        reporters : list[OpenMMReporter], optional
            Custom reporters to inject into the simulation, by default None.
        """
        # Copy the restart checkpoint to the output directory
        shutil.copy(self.checkpoint_file, self.restart_file)

        # Copy the static input files to the output directory
        if self.copy_input_files and self.top_file is not None:
            self.top_file = shutil.copy(self.top_file, self.output_dir)

        # Initialize an OpenMM simulation
        sim = self.config.configure_simulation(
            pdb_file=self.restart_file,
            top_file=self.top_file,
        )

        # Set up a reporter to write a simulation trajectory file
        sim.reporters.append(
            app.DCDReporter(
                file=self.trajectory_file,
                reportInterval=self.config.report_steps,
            ),
        )

        # Set up a reporter to write a simulation log file
        sim.reporters.append(
            app.StateDataReporter(
                file=str(self.log_file),
                reportInterval=self.config.report_steps,
                step=True,
                time=True,
                speed=True,
                potentialEnergy=True,
                temperature=True,
                totalEnergy=True,
            ),
        )

        # Inject the custom reporters
        if reporters is not None:
            sim.reporters.extend(reporters)

        # Attempt to locate a checkpoint file
        openmm_checkpoint = self.checkpoint_file.parent / 'seg.chk'

        # Load the checkpoint file (if it is a OpenMM checkpoint)
        if openmm_checkpoint.exists():
            sim.loadCheckpoint(str(openmm_checkpoint))

        # Run simulation
        sim.step(self.config.num_steps)

        # Save a checkpoint of the final state
        sim.saveCheckpoint(str(self.output_dir / 'seg.chk'))


class ContactMapRMSDReporter(OpenMMReporter):
    """Reporter to compute contact maps and RMSD from an OpenMM simulation."""

    def __init__(
        self,
        report_interval: int,
        reference_file: Path,
        cutoff_angstrom: float = 8.0,
        mda_selection: str = 'protein and name CA',
        openmm_selection: Sequence[str] = ('CA',),
    ) -> None:
        """Initialize the reporter.

        Parameters
        ----------
        report_interval : int
            The interval at which to write frames.
        reference_file : Path
            The reference PDB file for the analysis.
        cutoff_angstrom : float
            The angstrom cutoff distance for defining contacts
            (default is 8.0).
        mda_selection : str
            The MDAnalysis selection string for the atoms to use
            (default is 'protein and name CA').
        openmm_selection : Sequence[str]
            The OpenMM selection strings for the atoms to use
            (default is ('CA',)).
        """
        super().__init__(report_interval)
        self.cutoff_angstrom = cutoff_angstrom
        self.openmm_selection = openmm_selection

        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._rmsd: list[float] = []

        # Load the reference structure and save the positions
        mda_u = MDAnalysis.Universe(reference_file)
        self._ref = mda_u.select_atoms(mda_selection).positions.copy()

    def get_contact_maps(self) -> np.ndarray:
        """Get the contact maps from the simulation.

        Returns
        -------
        np.ndarray
            The contact maps from the simulation as a ragged array
            shaped as (n_frames, *).
        """
        # Concatenate the row and col indices into a single array
        contact_maps = [np.concatenate(x) for x in zip(self._rows, self._cols)]

        # Collect the contact maps in a ragged numpy array
        contact_maps = np.array(contact_maps, dtype=object)

        return contact_maps

    def get_rmsds(self) -> np.ndarray:
        """Get the RMSDs from the simulation.

        Returns
        -------
        np.ndarray
            The RMSDs from the simulation shaped as (n_frames, 1).
        """
        return np.array(self._rmsd).reshape(-1, 1)

    def report(self, simulation: app.Simulation, state: openmm.State) -> None:
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        # Get the atom indices for the selection
        atom_indices = [
            a.index
            for a in simulation.topology.atoms()
            if a.name in self.openmm_selection
        ]

        # Get the atomic coordinates of the selection
        positions = state.getPositions(asNumpy=True)
        positions = positions[atom_indices].astype(np.float32)

        # Convert positions from nanometers to angstroms
        positions *= 10.0

        # Compute the contact map
        contact_map = distances.contact_matrix(
            positions,
            self.cutoff_angstrom,
            returntype='sparse',
        )

        # Convert the contact map to sparse format
        coo_matrix = contact_map.tocoo()

        # Append the row and col indices to lists
        self._rows.append(coo_matrix.row.astype('int16'))
        self._cols.append(coo_matrix.col.astype('int16'))

        # Compute the RMSD
        rmsd = rms.rmsd(positions, self._ref, superposition=True)
        self._rmsd.append(rmsd)
