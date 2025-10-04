# BSK-RL: Basilisk + Reinforcement Learning Framework

## Overview

BSK-RL is a sophisticated Python package that creates high-fidelity Gymnasium environments for spacecraft tasking problems. It combines the Basilisk spacecraft simulation framework with reinforcement learning abstractions to enable research in autonomous spacecraft operations.

**Key Features:**
- High-fidelity spacecraft dynamics simulation
- Standard Gymnasium and PettingZoo interfaces for RL
- Support for single and multi-agent scenarios
- Modular architecture for easy customization
- Comprehensive examples and documentation

## Prerequisites

### System Requirements
- Python >= 3.10.0
- macOS, Linux, or Windows (with WSL recommended)
- At least 8GB RAM (16GB recommended for multi-agent scenarios)

### Required Dependencies
- Basilisk >= 2.8.9 (spacecraft simulation framework)
- Gymnasium (RL environment interface)
- PettingZoo >= 1.24.0 (multi-agent environments)
- NumPy, SciPy, Pandas

## Initial Setup

### 1. Create Virtual Environment (Recommended)
```bash
# Create a new virtual environment
python -m venv bsk_rl_env

# Activate the environment
# On macOS/Linux:
source bsk_rl_env/bin/activate
# On Windows:
# bsk_rl_env\Scripts\activate
```

### 2. Install BSK-RL
```bash
# Install from the cloned repository
cd /Users/zacharylee/nasa2025/bsk_rl
pip install -e .

# Or install with all optional dependencies
pip install -e ".[rllib,examples,dev]"
```

### 3. Verify Installation
```python
# Test basic import
python -c "import bsk_rl; print(f'BSK-RL version: {bsk_rl.__version__}')"

# Test environment creation
python -c "import gymnasium as gym; env = gym.make('GeneralSatelliteTasking-v1'); print('Environment created successfully!')"
```

## Quick Start Examples

### 1. Basic Single Satellite Environment
```python
import gymnasium as gym
from bsk_rl import sats, act, obs, scene

# Create a basic environment
env = gym.make(
    "SatelliteTasking-v1",
    satellite=sats.ImagingSatellite.default_sat(),
    actions=[act.Drift, act.Image],
    observations=[obs.Time, obs.SatProperties],
    scenario=scene.UniformTargets(n_targets=10),
)

# Run a simple episode
observation, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### 2. Multi-Agent Constellation
```python
from bsk_rl import ConstellationTasking

# Create a multi-agent environment
env = ConstellationTasking.from_default(n_sats=3)

# Run with random actions
observations, infos = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## Running Examples

The repository includes comprehensive Jupyter notebook examples in the `/examples/` directory:

### Basic Examples
1. **Simple Environment** (`simple_environment.ipynb`)
   ```bash
   jupyter notebook examples/simple_environment.ipynb
   ```
   Learn the basics of creating and interacting with environments.

2. **Multi-Agent Environments** (`multiagent_envs.ipynb`)
   ```bash
   jupyter notebook examples/multiagent_envs.ipynb
   ```
   Explore constellation coordination and multi-satellite missions.

### Advanced Examples
3. **RSO Inspection** (`rso_inspection.ipynb`)
   - Resident Space Object inspection missions
   - Relative motion and proximity operations

4. **Communication Actions** (`communication_action.ipynb`)
   - Inter-satellite communication strategies
   - Line-of-sight constraints and multi-hop relays

5. **RL Training** (`rllib_training.ipynb`)
   - Integration with Ray RLLib for distributed training
   - Requires: `pip install -e ".[rllib]"`

## Key Components

### Satellites (`bsk_rl.sats`)
- `Satellite`: Abstract base class
- `AccessSatellite`: Standard satellite with access calculations
- `ImagingSatellite`: Earth observation satellite

### Actions (`bsk_rl.act`)
**Discrete Actions:**
- `Drift`: Maintain current state
- `Charge`: Solar panel charging
- `Image`: Target imaging
- `Downlink`: Data transmission
- `Desat`: Momentum desaturation

**Continuous Actions:**
- `ImpulsiveThrust`: Direct thrust control
- `ImpulsiveThrustHill`: Hill frame thrust

### Observations (`bsk_rl.obs`)
- `Time`: Simulation time
- `SatProperties`: Spacecraft state (position, velocity, resources)
- `OpportunityProperties`: Upcoming targets
- `Eclipse`: Solar eclipse information

### Scenarios (`bsk_rl.scene`)
- `UniformTargets`: Random Earth targets
- `CityTargets`: Population-based targets
- `UniformNadirScanning`: Continuous scanning
- `SphericalRSO`: Space object inspection

### Rewards (`bsk_rl.data`)
- `UniqueImageReward`: First-time imaging rewards
- `ScanningTimeReward`: Time-based scanning
- `ResourceReward`: Resource optimization

## Common Use Cases

### Earth Observation Mission
```python
env = gym.make(
    "SatelliteTasking-v1",
    satellite=sats.ImagingSatellite(
        battery_capacity=5000,  # Wh
        data_capacity=1000,     # MB
        max_torque=1.0,        # Nm
    ),
    actions=[act.Drift, act.Charge, act.Image, act.Downlink],
    observations=[obs.Time, obs.SatProperties, obs.OpportunityProperties],
    scenario=scene.CityTargets(n_targets=20),
    rewarder=data.UniqueImageReward(reward=1.0),
)
```

### Space Debris Inspection
```python
env = gym.make(
    "SatelliteTasking-v1",
    satellite=sats.AccessSatellite(),
    actions=[act.Drift, act.ImpulsiveThrust],
    observations=[obs.Time, obs.SatProperties, obs.RelativeProperties],
    scenario=scene.SphericalRSO(n_rso=5),
    rewarder=data.RSOInspectionReward(),
)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=bsk_rl

# Run specific test categories
pytest tests/unittest/  # Unit tests only
pytest tests/integration/  # Integration tests only
```

## Documentation

### Local Documentation
```bash
# Build documentation locally
cd docs
make html

# View in browser
open _build/html/index.html  # macOS
# xdg-open _build/html/index.html  # Linux
```

### Online Resources
- [GitHub Repository](https://github.com/AVSLab/bsk_rl)
- [PyPI Package](https://pypi.org/project/bsk-rl/)
- Academic Publications (see README for citations)

## Development Workflow

### Code Style
The project uses:
- Black for formatting
- Ruff for linting
- isort for import sorting

```bash
# Format code
black src/ tests/
ruff check src/ tests/
isort src/ tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'Basilisk'**
   ```bash
   # Ensure Basilisk is installed
   pip install basilisk
   ```

2. **Environment Creation Failed**
   ```bash
   # Check Python version
   python --version  # Must be >= 3.10
   
   # Reinstall with dependencies
   pip install -e ".[examples]"
   ```

3. **Jupyter Notebooks Not Working**
   ```bash
   # Install notebook dependencies
   pip install jupyter matplotlib
   ```

### Performance Tips
- Use `n_procs=1` for debugging
- Increase `n_procs` for parallel environments
- Reduce `max_steps` for faster iterations
- Use smaller constellations for initial testing

## Next Steps

1. **Explore Examples**: Start with `simple_environment.ipynb`
2. **Read Documentation**: Check the API reference
3. **Customize Environments**: Create your own scenarios
4. **Train RL Agents**: Use RLLib or stable-baselines3
5. **Contribute**: Add new features or improvements

## Support

- **Issues**: [GitHub Issues](https://github.com/AVSLab/bsk_rl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AVSLab/bsk_rl/discussions)
- **Email**: See repository README for contact information

## License

MIT License - See LICENSE file for details

---

*This overview was generated for the BSK-RL repository at /Users/zacharylee/nasa2025/bsk_rl*