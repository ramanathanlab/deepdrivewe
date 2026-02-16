# Academy-Based Agentic Framework for deepdrivewe

This module provides an alternative implementation of deepdrivewe using the [Academy framework](https://docs.academy-agents.org/stable/) for federated actors and agents. It replaces the Colmena-based thinker-doer pattern with a multi-agent system where autonomous agents cooperate to run MD simulations and perform adaptive sampling.

## Architecture

The Academy-based implementation uses a hierarchical agent structure:

```
OrchestratorAgent (Workflow Coordinator)
├── SimulationPoolAgent (Worker Pool Manager)
│   ├── SimulationAgent (Worker 1)
│   ├── SimulationAgent (Worker 2)
│   └── SimulationAgent (Worker N)
└── EnsembleManagerAgent (Weighted Ensemble State Manager)
```

## Key Components

### Agents

- **OrchestratorAgent**: Coordinates the overall weighted ensemble workflow, advancing iterations and managing checkpoints
- **SimulationPoolAgent**: Manages a pool of simulation workers with load balancing and fault tolerance
- **SimulationAgent**: Executes individual MD simulations (currently supports OpenMM only)
- **EnsembleManagerAgent**: Manages weighted ensemble state, binning, resampling, and recycling

### Configuration

- **AcademyWorkflowConfig**: Top-level workflow configuration
- **SimulationPoolConfig**: Configuration for simulation pool and workers
- **AnalysisPoolConfig**: Configuration for analysis plugins (Phase 3)

## Academy Framework Patterns

### Actions

Actions are methods decorated with `@action` that can be invoked remotely via agent handles:

```python
@action
async def run_simulation(self, metadata: dict[str, Any]) -> dict[str, Any]:
    """Run an MD simulation."""
    # Implementation
```

### Loops

Loops are background tasks decorated with `@loop` that run continuously:

```python
@loop
async def load_balance(self, shutdown: asyncio.Event) -> None:
    """Distribute tasks across workers."""
    while not shutdown.is_set():
        # Implementation
```

### Communication

Agents communicate via handles using asynchronous message passing:

```python
# Get handle to another agent
result = await other_agent.some_action(param1, param2)
```

## Usage Example

See `examples/academy_workflow_example.py` for a complete example:

```python
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from deepdrivewe.academy_agents import OrchestratorAgent

async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:
    # Launch agents
    orchestrator = await manager.launch(OrchestratorAgent, ...)
    
    # Start workflow
    await orchestrator.start_workflow()
```

## Implementation Status

### Phase 1: Core Infrastructure ✅
- [x] Base agent class with logging and error handling
- [x] Configuration models for workflows
- [x] OrchestratorAgent with workflow coordination
- [x] EnsembleManagerAgent wrapping existing WE logic
- [x] SimulationAgent for running OpenMM simulations
- [x] SimulationPoolAgent with load balancing

### Phase 2: Simulation Pool ✅
- [x] Load balancing across multiple workers
- [x] Fault tolerance with automatic retry logic
- [x] Integration with OpenMMSimulation
- [x] Dynamic worker scaling interface (placeholder)

### Phase 3: Analysis Agents ✅
- [x] AnalysisPoolAgent interface
- [x] CVAE analyzer plugin
- [x] LOF analyzer plugin
- [-] ANCA analyzer plugin (not found in codebase, skipped)

### Phase 4: Goal-Oriented Rewards (Planned)
- [ ] Reward model framework
- [ ] Goal evaluation loop
- [ ] Adaptive sampling based on rewards

## Key Differences from Colmena Version

1. **Agent-based vs Queue-based**: Academy uses autonomous agents with direct communication instead of queue-based task distribution
2. **Asynchronous by default**: All agent actions are async, enabling better concurrency
3. **Distributed-first**: Academy is designed for distributed deployment from the ground up
4. **Type-safe communication**: Agent handles provide type-safe remote method invocation
5. **Pluggable exchanges**: Can use LocalExchangeFactory for testing or RedisExchangeFactory for distributed deployment

## Testing

Run the unit tests:

```bash
pytest tests/academy_agents/
```

## Future Enhancements

- Complete dynamic worker scaling implementation
- Add support for Amber and SynD simulation engines
- Implement analysis agent plugins (CVAE, ANCA, LOF)
- Add goal-oriented reward models
- Implement multi-view trajectory analysis
- Add distributed deployment examples with Redis exchange

