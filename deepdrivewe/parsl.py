"""Utilities to build Parsl configurations."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal
from typing import Sequence
from typing import Union

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, PBSProProvider
from pydantic import Field
from parsl.launchers import MpiExecLauncher

from deepdrivewe.api import BaseModel


class BaseComputeConfig(BaseModel, ABC):
    """Compute config (HPC platform, number of GPUs, etc)."""

    # Name of the platform to uniquely identify it
    name: Literal[''] = ''

    @abstractmethod
    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalConfig(BaseComputeConfig):
    """Local compute config."""

    name: Literal['local'] = 'local'  # type: ignore[assignment]

    max_workers: int = Field(
        default=1,
        description='Number of workers to use.',
    )
    cores_per_worker: float = Field(
        default=1.0,
        description='Number of cores per worker.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    label: str = Field(
        default='cpu_htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for local execution."""
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.label,
                    max_workers=self.max_workers,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationConfig(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal['workstation'] = 'workstation'  # type: ignore[assignment]

    available_accelerators: int | Sequence[str] = Field(
        default=1,
        description='Number of GPU accelerators to use.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )
    label: str = Field(
        default='gpu_htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for workstation execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class HybridWorkstationConfig(BaseComputeConfig):
    """Run simulations on CPU and AI models on GPU."""

    name: Literal['hybrid_workstation'] = 'hybrid_workstation'  # type: ignore[assignment]

    cpu_config: LocalConfig = Field(
        description='Config for the CPU executor to run simulations.',
    )
    gpu_config: WorkstationConfig = Field(
        description='Config for the GPU executor to run AI models.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for hybrid execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.gpu_config.retries,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.cpu_config.label,
                    max_workers=self.cpu_config.max_workers,
                    cores_per_worker=self.cpu_config.cores_per_worker,
                    worker_port_range=self.cpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address='localhost',
                    label=self.gpu_config.label,
                    cpu_affinity='block',
                    available_accelerators=self.gpu_config.available_accelerators,
                    worker_port_range=self.gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )

class SunspotSettings(BaseComputeConfig):
    """Configuration for running on Sunspot

    Each GPU tasks uses a single tile"""

    name: Literal["sunspot"] = "sunspot"  # type: ignore[assignment]
    label: str = 'htex'
    worker_init: str = ""

    num_nodes: int = 1
    """Number of nodes to request"""
    scheduler_options: str = ""
    account: str
    """The account to charge compute to."""
    queue: str
    """Which queue to submit jobs to, will usually be prod."""
    walltime: str
    """Maximum job time."""
    retries: int = 0
    """Number of retries upon failure."""
    cpus_per_node: int = 208
    strategy: str = "simple"

    def get_parsl_config(self, run_dir: str | Path)  -> Config:
        #accel_ids = [it for it in range(24)]
        if True:
            accel_string=""
            for gid in range(6):
               for tid in range(2):
                  accel_string += f"{gid}.{tid} "
            accel_ids = accel_string.split()
            with open("/home/avasan/config_log.log", "w") as f:
                f.write(f" Setting available accels to {accel_ids}")

        return Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    available_accelerators=accel_ids,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    #max_workers=12,
                    cores_per_worker=16,
                    heartbeat_period=30,
                    heartbeat_threshold=300,
                    worker_debug=False,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind",
                            overrides="--depth=208 --ppn 1"
                        ),  # Ensures 1 manger per node and allows it to divide work among all 208 threads
                        worker_init=self.worker_init,
                        nodes_per_block=self.num_nodes,
                        account=self.account,
                        queue=self.queue,
                        walltime=self.walltime,

                    ),
                ),
            ],
            run_dir=str(run_dir),
            checkpoint_mode='task_exit',
            retries=self.retries,
            app_cache=True,
        )


ComputeConfigTypes = Union[
    LocalConfig,
    WorkstationConfig,
    HybridWorkstationConfig,
    SunspotSettings,
]
