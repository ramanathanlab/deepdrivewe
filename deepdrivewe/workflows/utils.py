"""Workflow utils."""

from __future__ import annotations

import functools
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import TypeVar

from colmena.models import Result
from proxystore.proxy import Proxy
from proxystore.store import get_store
from proxystore.store.utils import ConnectorKeyT
from proxystore.store.utils import get_key

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')


class ProxyManager:
    """Manage the life cycle of a manually proxied object."""

    def __init__(self, store_name: str | None) -> None:
        """Initialize the proxy manager.

        Parameters
        ----------
        store_name: str | None
            The name of the store to use for proxying,
            or None if no store is available.
        """
        # Get the store
        if store_name is not None:
            self.store = get_store(store_name)

            # Raise an error if the store is not found
            if self.store is None:
                raise ValueError(f'Could not find store {store_name}')
        else:
            self.store = None

        # Store the key of the proxied object
        self.key: ConnectorKeyT | None = None

    def proxy(self, obj: T) -> T | Proxy[T]:
        """Proxy an object.

        Parameters
        ----------
        obj: T
            The object to proxy.

        Returns
        -------
        T Proxy[T]
            The proxied object or the original object if there is no store.
        """
        # If the store is not found, return the object
        if self.store is None:
            return obj

        # Proxy the object
        proxy_obj = self.store.proxy(obj)
        self.key = get_key(proxy_obj)
        return proxy_obj

    def evict(self) -> None:
        """Evict the proxied object."""
        if self.store is not None and self.key is not None:
            self.store.evict(self.key)
        self.key = None


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


def retry_on_exception(
    wait_time: int,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a function if an exception is raised.

    Parameters
    ----------
    wait_time: int
        Time to wait before retrying the function.
    """

    def decorator_retry(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper_retry(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                print(
                    f'Exception caught: {e}. \n'
                    f'Retrying after {wait_time} seconds...',
                )
                time.sleep(wait_time)
                return func(*args, **kwargs)

        return wrapper_retry

    return decorator_retry
