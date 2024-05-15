"""Run Amber simulations and analyze the results using cpptraj."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pydantic import BaseModel


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


class AmberConfig(BaseModel):
    """Config for an Amber simulation."""

    amber_exe: Path = field(
        metadata={
            'help': 'The path to the Amber executable.',
        },
    )
    md_input_file: Path = field(
        metadata={
            'help': 'The input file for the Amber simulation.',
        },
    )
    prmtop_file: Path = field(
        metadata={
            'help': 'The prmtop file for the Amber simulation.',
        },
    )
