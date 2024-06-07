"""Binning module for WESTPA."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np

from westpa_colmena.ensemble import SimMetadata
from westpa_colmena.resampling import Resampler


class Binner(ABC):
    """Binner for the progress coordinate."""

    @abstractmethod
    def assign(
        self,
        simulations: list[SimMetadata],
    ) -> list[list[SimMetadata]]:
        """Assign the simulations to bins."""
        ...


class RectilinearBinner:
    """Rectilinear binner for the progress coordinate."""

    def __init__(
        self,
        resampler: Resampler,
        bins: list[float],
        pcoord_idx: int = 0,
    ) -> None:
        """Initialize the binner.

        Parameters
        ----------
        resampler : Resampler
            The resampler to use for the binning.
        bins : list[float]
            The bin edges for the progress coordinate.
        pcoord_idx : int
            The index of the progress coordinate to use for binning.
            Default is 0.
        """
        self.resampler = resampler
        self.bins = bins
        self.pcoord_idx = pcoord_idx

    def assign(
        self,
        sims: list[SimMetadata],
    ) -> list[list[SimMetadata]]:
        """Bin the progress coordinate."""
        # Extract the progress coordinates
        pcoords = [sim.parent_pcoord[self.pcoord_idx] for sim in sims]

        # Bin the progress coordinates
        sim_assignments = np.digitize(pcoords, self.bins)

        # Assign the sims to the bins
        binned_sims: list[list[SimMetadata]] = [
            [] for _ in range(len(self.bins))
        ]

        # Assign the sims to the bins
        for sim, bin_index in zip(sims, sim_assignments):
            binned_sims[bin_index].append(sim)

        # Remove empty bins
        binned_sims = [i for i in binned_sims if i]

        return binned_sims
