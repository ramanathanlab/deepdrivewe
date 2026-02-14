# Academy-based NTL9 Protein Folding Example

This example demonstrates the complete Academy agents framework for weighted ensemble simulations of NTL9 protein folding using OpenMM and Huber-Kim resampling.

## Overview

This is an Academy-based reimplementation of the `openmm_ntl9_hk` example, replacing the Colmena-based workflow with Academy agents. It demonstrates:

- **OrchestratorAgent**: Coordinates the overall workflow
- **EnsembleManagerAgent**: Manages weighted ensemble state and resampling
- **SimulationPoolAgent**: Distributes simulations across workers
- **SimulationAgent**: Executes individual MD simulations

## Files

- `main_academy.py`: Main script using Academy agents
- `config_minimal.yaml`: Minimal test configuration (3 iterations, 2 workers)
- `README.md`: This file

## Configuration

The minimal test configuration (`config_minimal.yaml`) is designed for quick validation:

- **Iterations**: 3 (vs 106 in production)
- **Ensemble size**: 2 basis states (vs 4 in production)
- **Simulation length**: 1 ps (vs 10 ps in production)
- **Workers**: 2 simulation workers
- **Platform**: CPU (for portability)

## Running the Example

### Prerequisites

```bash
# Install deepdrivewe with Academy support
pip install -e .

# Ensure academy-py is installed
pip install academy-py
```

### Run the minimal test

```bash
# Set OpenMM to use single thread per simulation
export OPENMM_CPU_THREADS=1

# Run the Academy-based workflow
python examples/openmm_ntl9_hk_academy/main_academy.py \
    --config examples/openmm_ntl9_hk_academy/config_minimal.yaml
```

### Expected Output

The workflow will:

1. Initialize the weighted ensemble with 2 basis states
2. Launch 4 Academy agents (1 orchestrator, 1 ensemble manager, 1 pool, 2 workers)
3. Run 3 iterations of weighted ensemble simulation
4. Save checkpoints after each iteration
5. Log progress to `runs/ntl9-academy-test/runtime.log`

### Output Files

```
runs/ntl9-academy-test/
├── params.yaml              # Saved configuration
├── runtime.log              # Execution log
├── simulations/             # Simulation outputs
│   ├── iter_0000_walker_0000/
│   ├── iter_0000_walker_0001/
│   └── ...
└── checkpoints/             # Ensemble checkpoints
    ├── checkpoint_iter_0001.h5
    ├── checkpoint_iter_0002.h5
    └── checkpoint_iter_0003.h5
```

## Comparison with Colmena Version

### Colmena Version (`openmm_ntl9_hk/main.py`)

- Uses Colmena's `PipeQueues` for task distribution
- Uses `ParslTaskServer` for execution
- Uses `WESTPAThinker` for workflow logic
- Requires ProxyStore for data management

### Academy Version (`openmm_ntl9_hk_academy/main_academy.py`)

- Uses Academy's `Manager` and agent handles
- Uses Academy's `@action` and `@loop` decorators
- Distributed agent architecture (Orchestrator, EnsembleManager, SimulationPool, SimulationAgent)
- Native async/await patterns
- No external queue or proxy store needed

## Validation Criteria

✅ **Success Criteria**:
- All agents launch successfully
- Simulations execute without errors
- Ensemble state advances through iterations
- Checkpoints are saved correctly
- All agents shut down cleanly

❌ **Failure Indicators**:
- Agent launch timeouts
- Simulation execution errors
- Checkpoint save/load failures
- Agent communication errors

## Troubleshooting

### Issue: Agents hang on launch
**Solution**: Ensure `ThreadPoolExecutor` is passed to `Manager.from_exchange_factory()`

### Issue: Simulations fail
**Solution**: Check that OpenMM is installed and CPU platform is available

### Issue: Checkpoint errors
**Solution**: Ensure output directory has write permissions

## Next Steps

After successful validation:

1. Run with production configuration (106 iterations, 4 basis states)
2. Test with GPU platform for faster simulations
3. Scale to distributed deployment with RedisExchangeFactory
4. Add analysis agents for real-time monitoring

## References

- Original example: `examples/openmm_ntl9_hk/`
- Academy documentation: https://docs.academy-agents.org/
- DeepDriveWE paper: [Link to paper]

