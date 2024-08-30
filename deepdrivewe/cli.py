"""CLI for the deepdrivewe package."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def version() -> None:
    """Print the version of deepdrivewe."""
    from deepdrivewe import __version__

    print(f'deepdrivewe, version {__version__}')


@app.command()
def to_pdb(
    coordinate_file: Path = typer.Option(  # noqa: B008
        ...,
        '--coordinate_file',
        '-c',
        help='Path to the input coordinate file (e.g., .rst, etc).',
    ),
    top_file: Path = typer.Option(  # noqa: B008
        ...,
        '--top_file',
        '-t',
        help='Path to the input topology file (e.g., .prmtop, .top, etc).',
    ),
    output_pdb_file: Path = typer.Option(  # noqa: B008
        ...,
        '--output_pdb_file',
        '-o',
        help='Path to the output PDB file.',
    ),
) -> None:
    """Convert a prmtop and rst7 file to a PDB file using MDTraj."""
    import mdtraj

    # Load the topology and trajectory
    trajectory: mdtraj.Trajectory = mdtraj.load(coordinate_file, top=top_file)

    # Save the trajectory to a PDB file
    trajectory.save(output_pdb_file)

    # Print the path to the output PDB file
    print(f'PDB file saved to {output_pdb_file}')


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
