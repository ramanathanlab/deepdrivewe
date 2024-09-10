"""Utility functions for the AI module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


# TODO: We are not checkpointing the latent space history. This is
# may cause an issue if we find the latent history is very important
# for convergence and we have a restart. We should consider checkpointing.
class LatentSpaceHistory:
    """A class to store the latent space history."""

    def __init__(self) -> None:
        """Initialize the latent space history."""
        # A (n, d) array where n is the number of frames and d is
        # the latent space dimensionality
        self.z = np.array([])
        # A (n, 1) array of progress coordinates for each frame
        self.pcoords = np.array([])

    def __bool__(self) -> bool:
        """Return True if the history is not empty."""
        return bool(len(self.z))

    def update(self, z: ArrayLike, pcoords: ArrayLike) -> None:
        """Update the latent space history.

        Parameters
        ----------
        z: array-like
            The latent space coordinates (n_frames, d)
        pcords: array-like
            The progress coordinates (n_frames, 1)
        """
        self.z = z
        self.pcoords = pcoords

    def plot(
        self,
        output_path: Path,
        color: ArrayLike | None = None,
        cblabel: str = 'Progress Coordinate',
        title: str = '',
    ) -> None:
        """Create a scatter plot.

        Parameters
        ----------
        x: array-like
            X data for the scatter plot
        y: array-like
            Y data for the scatter plot
        color: array-like
            Data used for coloring the points
        xlabel: str
            Label for the X-axis
        ylabel: str
            Label for the Y-axis
        title: str
            Title of the plot
        """
        import matplotlib.pyplot as plt

        # Set the color data to the progress coordinates if not provided
        color = self.pcoords if color is None else color

        print(
            f'Plotting latent space to with {len(self.z)} '
            f'and color with shape {len(color)} frames to {output_path}',
        )

        # Create the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            xs=self.z[:, 0],
            ys=self.z[:, 1],
            zs=self.z[:, 2],
            c=color,
            cmap='viridis',
        )
        fig.colorbar(scatter, label=cblabel)
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.set_zlabel(r'$z_3$')
        ax.set_title(title)
        # plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.savefig(output_path)
        plt.close()
