"""Workflow utils."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from colmena.models import Result


class ResultLogger:
    """Logger for results."""

    def __init__(self, result_dir: Path) -> None:
        """Initialize the result logger.

        Parameters
        ----------
        result_dir: Path
            Directory in which to store outputs
        """
        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir

        # Number of times a given task has been submitted
        self.task_counter: defaultdict[str, int] = defaultdict(int)

    def log(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        # Increment the task counter
        self.task_counter[topic] += 1

        # Write the result to a jsonl file
        with open(self.result_dir / f'{topic}.json', 'a') as f:
            print(result.json(exclude={'inputs', 'value'}), file=f)
