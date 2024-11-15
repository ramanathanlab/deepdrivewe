"""CLI for the deepdrivewe package."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print
from rich.console import Console

app = typer.Typer()


@app.command()
def version() -> None:
    """Print the version of deepdrivewe."""
    from deepdrivewe import __version__

    print(f'deepdrivewe, version {__version__}')


@app.command()
def print_errors(
    run_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--run_dir',
        '-r',
        help='Path to the run directory.',
    ),
) -> None:
    """Parse the task result files and print any errors."""
    console = Console()

    # Find all the task result files
    results_dir = run_dir / 'results'

    # Read the simulation, train, and inference results
    for file_path in results_dir.glob('*.json'):
        # Read the entire file as text
        file_text = file_path.read_text()

        # Parse each line as JSON
        for line in file_text.splitlines():
            data = json.loads(line)
            if 'failure_info' in data and 'traceback' in data['failure_info']:
                error_message = data['failure_info']['traceback']
                console.print(
                    f"[bold red]Task ID:[/bold red] {data['task_id']}",
                )
                console.print(
                    f"[bold yellow]Method:[/bold yellow] {data['method']}",
                )
                console.print('[bold blue]Traceback:[/bold blue]\n')
                console.print(error_message, style='red')


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
