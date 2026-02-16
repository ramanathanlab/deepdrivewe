"""Unit tests for analysis agents and analyzer plugins."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest


def test_analysis_imports() -> None:
    """Test that analysis agent modules can be imported."""
    from deepdrivewe.academy_agents import AnalysisPoolAgent
    from deepdrivewe.academy_agents import AnalyzerPlugin
    from deepdrivewe.academy_agents import CVAEAnalyzer
    from deepdrivewe.academy_agents import LOFAnalyzer

    assert AnalysisPoolAgent is not None
    assert AnalyzerPlugin is not None
    assert CVAEAnalyzer is not None
    assert LOFAnalyzer is not None


def test_analysis_pool_config(tmp_path: Path) -> None:
    """Test that AnalysisPoolConfig can be created."""
    from deepdrivewe.academy_agents import AnalysisPoolConfig

    config = AnalysisPoolConfig(
        output_dir=tmp_path / 'analysis',
        enabled_analyzers=['cvae', 'lof'],
        analyzer_configs={
            'cvae': {
                'cvae_config': {
                    'latent_dim': 3,
                    'epochs': 10,
                },
            },
            'lof': {
                'n_neighbors': 20,
                'metric': 'cosine',
            },
        },
    )

    assert config.output_dir == tmp_path / 'analysis'
    assert 'cvae' in config.enabled_analyzers
    assert 'lof' in config.enabled_analyzers
    assert config.analyzer_configs['cvae']['cvae_config']['latent_dim'] == 3


def test_cvae_analyzer_creation(tmp_path: Path) -> None:
    """Test that CVAEAnalyzer can be instantiated."""
    from deepdrivewe.academy_agents import CVAEAnalyzer

    config = {
        'cvae_config': {
            'latent_dim': 3,
            'epochs': 10,
            'device': 'cpu',
        },
    }

    analyzer = CVAEAnalyzer(
        config=config,
        output_dir=tmp_path / 'cvae',
    )

    assert analyzer.get_name() == 'cvae'
    assert analyzer.output_dir.exists()
    assert analyzer.model is not None


def test_lof_analyzer_creation(tmp_path: Path) -> None:
    """Test that LOFAnalyzer can be instantiated."""
    from deepdrivewe.academy_agents import LOFAnalyzer

    config = {
        'n_neighbors': 20,
        'metric': 'cosine',
    }

    analyzer = LOFAnalyzer(
        config=config,
        output_dir=tmp_path / 'lof',
    )

    assert analyzer.get_name() == 'lof'
    assert analyzer.output_dir.exists()
    assert analyzer.n_neighbors == 20
    assert analyzer.metric == 'cosine'


def test_analysis_pool_agent_creation(tmp_path: Path) -> None:
    """Test that AnalysisPoolAgent can be instantiated."""
    from deepdrivewe.academy_agents import AnalysisPoolAgent

    agent = AnalysisPoolAgent(
        output_dir=tmp_path / 'analysis',
        enabled_analyzers=['cvae', 'lof'],
        analyzer_configs={
            'cvae': {
                'cvae_config': {
                    'latent_dim': 3,
                    'device': 'cpu',
                },
            },
            'lof': {
                'n_neighbors': 20,
            },
        },
    )

    assert agent.output_dir.exists()
    assert 'cvae' in agent.analyzers
    assert 'lof' in agent.analyzers
    assert len(agent.analyzers) == 2


@pytest.mark.asyncio
async def test_lof_analyzer_compute_lof(tmp_path: Path) -> None:
    """Test that LOFAnalyzer can compute LOF scores."""
    from deepdrivewe.academy_agents import LOFAnalyzer

    config = {
        'n_neighbors': 5,
        'metric': 'euclidean',
    }

    analyzer = LOFAnalyzer(
        config=config,
        output_dir=tmp_path / 'lof',
    )

    # Create synthetic feature data
    features = np.random.rand(10, 3)

    # Compute LOF scores
    lof_scores = analyzer._compute_lof(features)

    assert lof_scores.shape == (10,)
    assert np.all(lof_scores <= 0)  # LOF scores are negative

