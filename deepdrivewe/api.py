"""API for running workflows with Colmena."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

import yaml  # type: ignore[import-untyped]
from colmena.models import Result
from pydantic import BaseModel as _BaseModel

T = TypeVar('T')


class BaseModel(_BaseModel):
    """Provide an easy interface to read/write YAML files."""

    def dump_yaml(self, filename: str | Path) -> None:
        """Dump settings to a YAML file."""
        with open(filename, mode='w') as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: type[T], filename: str | Path) -> T:
        """Load settings from a YAML file."""
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


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
