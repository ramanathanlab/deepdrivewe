"""Analysis agents for trajectory analysis and ML-based adaptive sampling."""

from __future__ import annotations

import asyncio
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from academy.agent import action
from academy.handle import Handle

from deepdrivewe.academy_agents.base import AcademyAgent


class AnalyzerPlugin(ABC):
    """Base class for analyzer plugins.

    Analyzer plugins provide specialized analysis capabilities such as
    CVAE latent space projection, LOF anomaly detection, or ANCA analysis.
    Each plugin implements a common interface for processing simulation data.
    """

    @abstractmethod
    async def analyze(
        self,
        sim_results: list[dict[str, Any]],
        iteration_id: int,
    ) -> dict[str, Any]:
        """Analyze simulation results.

        Parameters
        ----------
        sim_results : list[dict[str, Any]]
            List of simulation results containing trajectory data.
        iteration_id : int
            Current iteration number.

        Returns
        -------
        dict[str, Any]
            Analysis results containing computed features, scores, or embeddings.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get the analyzer name.

        Returns
        -------
        str
            Name of the analyzer (e.g., 'cvae', 'lof', 'anca').
        """
        ...


class CVAEAnalyzer(AnalyzerPlugin):
    """CVAE analyzer for computing latent space embeddings.

    This analyzer uses a Convolutional Variational Autoencoder to project
    contact maps into a low-dimensional latent space for visualization and
    adaptive sampling.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Initialize the CVAE analyzer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing CVAE parameters.
        output_dir : Path
            Directory to store analysis outputs.
        """
        from deepdrivewe.ai import ConvolutionalVAE
        from deepdrivewe.ai import ConvolutionalVAEConfig
        from deepdrivewe.ai.utils import LatentSpaceHistory

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load CVAE configuration
        cvae_config = ConvolutionalVAEConfig(**config.get('cvae_config', {}))
        checkpoint_path = config.get('checkpoint_path')

        # Initialize CVAE model
        self.model = ConvolutionalVAE(
            config=cvae_config,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )

        # Initialize latent space history for tracking
        self.history = LatentSpaceHistory()

    async def analyze(
        self,
        sim_results: list[dict[str, Any]],
        iteration_id: int,
    ) -> dict[str, Any]:
        """Compute latent space embeddings using CVAE.

        Parameters
        ----------
        sim_results : list[dict[str, Any]]
            List of simulation results with 'contact_maps' in data field.
        iteration_id : int
            Current iteration number.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'latent_coords' (n_sims, latent_dim) array.
        """
        # Extract contact maps from simulation results
        contact_maps = [
            sim['data']['contact_maps'][-1] for sim in sim_results
        ]

        # Convert to int16 for memory efficiency
        contact_maps = [x.astype(np.int16) for x in contact_maps]

        # Run CVAE prediction in thread pool to avoid blocking
        latent_coords = await asyncio.to_thread(
            self.model.predict,
            x=contact_maps,
        )

        # Update history
        pcoords = np.array([
            sim['metadata']['pcoord'][-1][0] for sim in sim_results
        ])

        if self.history:
            latent_coords_full = np.concatenate([self.history.z, latent_coords])
            pcoords_full = np.concatenate([self.history.pcoords, pcoords])
        else:
            latent_coords_full = latent_coords
            pcoords_full = pcoords

        self.history.update(latent_coords_full, pcoords_full)

        # Save visualization
        output_path = self.output_dir / f'iteration_{iteration_id:06d}_latent.png'
        await asyncio.to_thread(self.history.plot, output_path)

        return {
            'latent_coords': latent_coords.tolist(),
            'latent_dim': latent_coords.shape[1],
        }

    def get_name(self) -> str:
        """Get analyzer name."""
        return 'cvae'


class LOFAnalyzer(AnalyzerPlugin):
    """LOF (Local Outlier Factor) analyzer for anomaly detection.

    This analyzer computes LOF scores in latent space to identify outlier
    simulations for adaptive sampling.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Initialize the LOF analyzer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary containing LOF parameters.
        output_dir : Path
            Directory to store analysis outputs.
        """
        from sklearn.neighbors import LocalOutlierFactor

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LOF configuration
        self.n_neighbors = config.get('n_neighbors', 20)
        self.metric = config.get('metric', 'cosine')

        # Initialize LOF model
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )

    async def analyze(
        self,
        sim_results: list[dict[str, Any]],
        iteration_id: int,
    ) -> dict[str, Any]:
        """Compute LOF scores for simulations.

        Parameters
        ----------
        sim_results : list[dict[str, Any]]
            List of simulation results. Expects 'latent_coords' in metadata
            if available, otherwise uses progress coordinates.
        iteration_id : int
            Current iteration number.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'lof_scores' array.
        """
        # Try to get latent coordinates from metadata (if CVAE ran first)
        # Otherwise fall back to progress coordinates
        if 'latent_coords' in sim_results[0].get('analysis', {}):
            features = np.array([
                sim['analysis']['latent_coords'] for sim in sim_results
            ])
        else:
            # Use progress coordinates as features
            features = np.array([
                sim['metadata']['pcoord'][-1] for sim in sim_results
            ])

        # Compute LOF scores in thread pool
        lof_scores = await asyncio.to_thread(
            self._compute_lof,
            features,
        )

        return {
            'lof_scores': lof_scores.tolist(),
        }

    def _compute_lof(self, features: np.ndarray) -> np.ndarray:
        """Compute LOF scores (blocking operation).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            LOF scores (n_samples,).
        """
        self.lof_model.fit(features)
        return self.lof_model.negative_outlier_factor_

    def get_name(self) -> str:
        """Get analyzer name."""
        return 'lof'


class AnalysisPoolAgent(AcademyAgent):
    """Agent that manages analysis tasks and routes them to analyzer plugins.

    This agent coordinates multiple analyzer plugins (CVAE, LOF, ANCA) and
    provides load balancing and fault tolerance for analysis tasks.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration for the analysis pool.
    analyzers : dict[str, AnalyzerPlugin]
        Dictionary of analyzer plugins keyed by name.
    """

    def __init__(
        self,
        output_dir: Path,
        enabled_analyzers: list[str],
        analyzer_configs: dict[str, Any],
    ) -> None:
        """Initialize the analysis pool agent.

        Parameters
        ----------
        output_dir : Path
            Directory to store analysis outputs.
        enabled_analyzers : list[str]
            List of enabled analyzer names (e.g., ['cvae', 'lof']).
        analyzer_configs : dict[str, Any]
            Configuration for each analyzer, keyed by analyzer name.
        """
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzer plugins
        self.analyzers: dict[str, AnalyzerPlugin] = {}

        for analyzer_name in enabled_analyzers:
            if analyzer_name == 'cvae':
                self.analyzers['cvae'] = CVAEAnalyzer(
                    config=analyzer_configs.get('cvae', {}),
                    output_dir=output_dir / 'cvae',
                )
            elif analyzer_name == 'lof':
                self.analyzers['lof'] = LOFAnalyzer(
                    config=analyzer_configs.get('lof', {}),
                    output_dir=output_dir / 'lof',
                )
            else:
                self.logger.warning(
                    f'Unknown analyzer: {analyzer_name}, skipping',
                )

        self.logger.info(
            f'Initialized AnalysisPoolAgent with analyzers: '
            f'{list(self.analyzers.keys())}',
        )

    @action
    async def analyze_simulations(
        self,
        sim_results: list[dict[str, Any]],
        iteration_id: int,
    ) -> dict[str, Any]:
        """Run all enabled analyzers on simulation results.

        Parameters
        ----------
        sim_results : list[dict[str, Any]]
            List of simulation results to analyze.
        iteration_id : int
            Current iteration number.

        Returns
        -------
        dict[str, Any]
            Combined analysis results from all analyzers.
        """
        self._log_action(
            'analyze_simulations',
            num_sims=len(sim_results),
            iteration=iteration_id,
        )

        analysis_results: dict[str, Any] = {}

        # Run analyzers in sequence (CVAE first, then LOF can use latent coords)
        # CVAE analyzer
        if 'cvae' in self.analyzers:
            try:
                cvae_results = await self.analyzers['cvae'].analyze(
                    sim_results,
                    iteration_id,
                )
                analysis_results['cvae'] = cvae_results

                # Add latent coords to sim_results for downstream analyzers
                for i, sim in enumerate(sim_results):
                    if 'analysis' not in sim:
                        sim['analysis'] = {}
                    sim['analysis']['latent_coords'] = cvae_results[
                        'latent_coords'
                    ][i]

                self.logger.info('CVAE analysis completed successfully')
            except Exception as e:
                self.logger.error(f'CVAE analysis failed: {e}')
                analysis_results['cvae'] = {'error': str(e)}

        # LOF analyzer
        if 'lof' in self.analyzers:
            try:
                lof_results = await self.analyzers['lof'].analyze(
                    sim_results,
                    iteration_id,
                )
                analysis_results['lof'] = lof_results

                # Add LOF scores to sim_results
                for i, sim in enumerate(sim_results):
                    if 'analysis' not in sim:
                        sim['analysis'] = {}
                    sim['analysis']['lof_score'] = lof_results['lof_scores'][i]

                self.logger.info('LOF analysis completed successfully')
            except Exception as e:
                self.logger.error(f'LOF analysis failed: {e}')
                analysis_results['lof'] = {'error': str(e)}

        return analysis_results

    @action
    async def get_status(self) -> dict[str, Any]:
        """Get the status of the analysis pool.

        Returns
        -------
        dict[str, Any]
            Status information including enabled analyzers.
        """
        return {
            'enabled_analyzers': list(self.analyzers.keys()),
            'num_analyzers': len(self.analyzers),
        }

