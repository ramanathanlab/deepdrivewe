"""Academy-based agentic framework for deepdrivewe.

This module provides an alternative implementation of deepdrivewe using
the Academy framework for federated actors and agents. It replaces the
Colmena-based thinker-doer pattern with a multi-agent system where
autonomous agents cooperate to run MD simulations and perform adaptive
sampling using ML-driven analysis.

Key Components
--------------
- OrchestratorAgent: Coordinates the overall workflow
- SimulationPoolAgent: Manages a pool of simulation workers
- SimulationAgent: Executes individual MD simulations
- EnsembleManagerAgent: Manages weighted ensemble state and resampling
- AnalysisPoolAgent: Routes analysis requests to specialized analyzers

Example
-------
>>> from academy.manager import Manager
>>> from academy.exchange import LocalExchangeFactory
>>> from deepdrivewe.academy_agents import OrchestratorAgent
>>>
>>> async with await Manager.from_exchange_factory(
...     factory=LocalExchangeFactory(),
... ) as manager:
...     orchestrator = await manager.launch(OrchestratorAgent, config=config)
...     await orchestrator.start_workflow()
"""

from __future__ import annotations

from deepdrivewe.academy_agents.base import AcademyAgent
from deepdrivewe.academy_agents.config import AcademyWorkflowConfig
from deepdrivewe.academy_agents.config import AnalysisPoolConfig
from deepdrivewe.academy_agents.config import SimulationPoolConfig
from deepdrivewe.academy_agents.ensemble import EnsembleManagerAgent
from deepdrivewe.academy_agents.orchestrator import OrchestratorAgent
from deepdrivewe.academy_agents.simulation import SimulationAgent
from deepdrivewe.academy_agents.simulation import SimulationPoolAgent

__all__ = [
    'AcademyAgent',
    'AcademyWorkflowConfig',
    'AnalysisPoolConfig',
    'EnsembleManagerAgent',
    'OrchestratorAgent',
    'SimulationAgent',
    'SimulationPoolAgent',
    'SimulationPoolConfig',
]

