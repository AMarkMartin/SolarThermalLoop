# Modular Solar Thermal System Architecture

## Overview

This system demonstrates a **component-based architecture** where each physical unit operation can be independently upgraded while maintaining integration with other components.

## Design Principles

### 1. Modularity
- Each component (solar panel, tank, load, valve, pump) is a self-contained class
- Components expose well-defined interfaces via `update()` and `get_state()` methods
- Physics models can be upgraded within components without affecting the system

### 2. Separation of Concerns
- **Components**: Handle physics and thermodynamics
- **Controllers**: Handle decision-making and optimization
- **System**: Orchestrates component interactions

### 3. Upgradability
- Swap physics models without changing integration code
- Example: Upgrade tank from fully-mixed to stratified model
- Example: Replace basic controller with MPC

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Environment                          │
│              (Solar, Ambient Temperature)               │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Controller                          │
│  (Monitors state, computes control actions)             │
└─────────────────────────────────────────────────────────┘
          │                                  │
          ▼                                  ▼
    ┌─────────┐                        ┌─────────┐
    │  Pump   │                        │  Pump   │
    │ (Solar) │                        │ (Load)  │
    └─────────┘                        └─────────┘
          │                                  │
          ▼                                  ▼
    ┌─────────┐                        ┌─────────┐
    │  Valve  │                        │  Valve  │
    │ (Solar) │                        │ (Load)  │
    └─────────┘                        └─────────┘
          │                                  │
          ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│ Solar Collector  │──────────────→│  Storage Tank   │
│  (Heat Source)   │              │  (Heat Storage) │
└──────────────────┘              └──────────────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  Thermal Load    │
                                  │  (Heat Demand)   │
                                  └──────────────────┘
```

## Component Interfaces

### Base Component Class

All components inherit from `Component` and implement:

```python
class Component(ABC):
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Update component state and return outputs"""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return current state for monitoring"""
        pass
```

### Key Components

#### 1. SolarCollector (FluidComponent)
**Inputs:**
- `T_inlet`: Fluid inlet temperature
- `flow_rate`: Mass flow rate
- `irradiance`: Solar irradiance
- `T_ambient`: Ambient temperature

**Outputs:**
- `T_outlet`: Fluid outlet temperature
- `Q_collected`: Heat power collected
- `T_collector`: Collector surface temperature

**Upgradable Parameters:**
- Collector efficiency model
- Heat loss coefficients
- Thermal mass effects
- Angle-dependent performance (IAM)

#### 2. StorageTank (FluidComponent)
**Inputs:**
- `T_inlet_solar`: Hot fluid from collector
- `flow_rate_solar`: Solar loop flow rate
- `T_inlet_load`: Return from load
- `flow_rate_load`: Load loop flow rate

**Outputs:**
- `T_tank`: Tank temperature
- `T_outlet_solar`: Return to collector
- `T_outlet_load`: Supply to load
- `Q_solar`, `Q_load`, `Q_loss`: Heat flows

**Upgradable Parameters:**
- Stratification model (single vs multi-node)
- Heat exchanger effectiveness
- Tank geometry effects

#### 3. ThermalLoad (FluidComponent)
**Inputs:**
- `T_inlet`: Supply temperature
- `flow_rate`: Flow rate
- `time_hours`: Current time (for scheduled loads)

**Outputs:**
- `T_outlet`: Return temperature
- `Q_demand`: Required heat power
- `Q_actual`: Actual heat delivered
- `load_fraction`: Fraction of demand met

**Upgradable Parameters:**
- Load profile model (constant, variable, scheduled)
- Building thermal model
- Weather-dependent loads

#### 4. Pump (Component)
**Inputs:**
- `speed`: Commanded speed (0-1)

**Outputs:**
- `flow_rate`: Resulting flow rate

**Upgradable Parameters:**
- Pump curves (head vs flow)
- Efficiency model
- Power consumption

#### 5. Valve (Component)
**Inputs:**
- `position`: Commanded position (0-1)

**Outputs:**
- `flow_rate`: Resulting flow rate

**Upgradable Parameters:**
- Valve characteristics (linear, equal-percentage)
- Pressure drop calculations

## Control System

### BasicController
- Differential temperature control for solar loop
- Load-based flow modulation
- Hysteresis to prevent cycling
- Tank over-temperature protection

### Future Controllers (Templates Provided)
- **PIDController**: For precise temperature control
- **OptimizingController**: MPC with weather forecasting

## System Integration

The `ModularThermalSystem` class:
1. Instantiates all components
2. Manages data flow between components
3. Coordinates timestep updates
4. Logs system performance

### Simulation Loop
```python
for each timestep:
    1. Get system state from all components
    2. Controller computes control actions
    3. Update pumps/valves based on control
    4. Update solar collector (inputs from tank, environment)
    5. Update thermal load (inputs from tank, time)
    6. Update storage tank (inputs from collector and load)
    7. Log results
```

## Upgrading Components

### Example: Upgrading Storage Tank to Stratified Model

**Current**: Single-node (fully mixed) tank
```python
class StorageTank(FluidComponent):
    def __init__(self, ...):
        self.T_tank = initial_temp  # Single temperature
```

**Upgraded**: Multi-node stratified tank
```python
class StratifiedTank(FluidComponent):
    def __init__(self, ..., num_nodes=10):
        self.T_nodes = np.ones(num_nodes) * initial_temp  # Temperature array
        self.node_height = params.height / num_nodes

    def update(self, dt, inputs):
        # Solve 1D heat diffusion equation
        # Handle inlet/outlet at specific nodes
        # Natural convection between nodes
        ...
```

**Integration**: Just swap the component!
```python
# Before
tank = StorageTank("Tank", params)

# After
tank = StratifiedTank("Tank", params, num_nodes=10)
# System code unchanged!
```

### Example: Upgrading Controller

**Current**: Rule-based differential temperature control

**Upgraded**: Model Predictive Control (MPC)
```python
class MPCController(Controller):
    def __init__(self, prediction_horizon=24):
        self.horizon = prediction_horizon

    def compute_control(self, system_state):
        # Get weather forecast
        # Predict load demand
        # Optimize control over horizon
        # Minimize cost function
        return optimal_control_actions
```

## Benefits of This Architecture

1. **Maintainability**: Each component is in its own module
2. **Testability**: Components can be unit tested independently
3. **Reusability**: Components can be used in different system configurations
4. **Scalability**: Easy to add new components (e.g., backup heater, dump load)
5. **Validation**: Physics models can be validated component-by-component
6. **Collaboration**: Different team members can work on different components

## File Structure

```
PassiveLogicInterview/
├── components.py           # All physical component classes
├── controller.py           # Control algorithms
├── modular_simulation.py   # System integration and simulation
├── ARCHITECTURE.md         # This document
└── archive/                # Version history
    └── v1.0_baseline/      # Original monolithic version
```

## Current Limitations & Future Improvements

### Known Issues
- Tank temperature can exceed realistic limits (needs improved control)
- No pressure drop calculations
- No pump power consumption tracking
- Simple solar model (no angle effects)

### Easy Fixes (Thanks to Modularity!)
- Add dump load component for overheating protection
- Upgrade controller with better logic
- Add more sophisticated load profiles
- Implement stratified tank model
- Add weather data integration

### Future Enhancements
- Hydraulic network solver for pressure/flow
- Auxiliary heating (backup boiler)
- Multiple storage tanks in series/parallel
- Heat exchangers (indirect systems)
- Economic optimization (utility rates)

## Example Usage

```python
# Create components with desired physics models
collector = SolarCollector("SC1", SolarCollectorParams(area=3.0))
tank = StorageTank("Tank", StorageTankParams(volume=0.3))
load = ThermalLoad("Load", ThermalLoadParams(load_profile='scheduled'))
controller = BasicController()

# Assemble system
system = ModularThermalSystem(collector, tank, load, controller)

# Run simulation
system.run_simulation(duration_hours=48.0)
system.plot_results()

# Later: upgrade any component without touching system code!
```

## Validation

Each component should be validated independently:
- **Solar Collector**: Compare to ASHRAE 93 test data
- **Storage Tank**: Verify energy balance, compare stratification to CFD
- **Load**: Match real building load profiles
- **Controller**: Test response to disturbances

System-level validation:
- Energy balance closure (< 1% error)
- Realistic temperature ranges
- Comparison to measured system performance
