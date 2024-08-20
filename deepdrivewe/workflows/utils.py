"""Workflow utils."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from colmena.models import Result

T = TypeVar('T')


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


def batch_data(data: list[T], batch_size: int) -> list[list[T]]:
    """Batch `data` into batches of size `batch_size`.

    Parameters
    ----------
    data: list[T]
        The data to batch.
    batch_size: int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The data batched into `batch_size` batches.
    """
    batches = [
        data[i * batch_size : (i + 1) * batch_size]
        for i in range(0, len(data) // batch_size)
    ]
    # Handle the leftover data if the batch size does not
    # divide the data evenly
    if len(data) > batch_size * len(batches):
        batches.append(data[len(batches) * batch_size :])

    return batches
