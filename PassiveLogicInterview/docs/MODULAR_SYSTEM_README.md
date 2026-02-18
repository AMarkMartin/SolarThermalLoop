# Modular Solar Thermal System with Thermal Load and Control

## Overview

This is an upgraded version of the baseline solar thermal simulation featuring:
- **Modular component architecture** - Each unit operation can be independently upgraded
- **Thermal load integration** - Continuous heat demand from storage
- **Active control system** - Automated pump and valve control
- **Valve components** - Flow control elements
- **Extensibility** - Easy to add new components or upgrade existing ones

## Key Features

### 1. Component-Based Architecture

Each physical component is a self-contained class with well-defined interfaces:

| Component | Purpose | Upgradeable Aspects |
|-----------|---------|---------------------|
| `SolarCollector` | Absorbs solar energy | Efficiency model, IAM, heat loss |
| `StorageTank` | Stores thermal energy | Stratification, geometry, insulation |
| `ThermalLoad` | Heat demand/consumption | Load profiles, building model |
| `Pump` | Fluid circulation | Pump curves, efficiency, power |
| `Valve` | Flow control | Characteristics, pressure drop |

### 2. Control System

Controllers monitor system state and command actuators:
- **BasicController**: Rule-based differential temperature control
- **ImprovedController**: Enhanced safety and adaptive features
- **PIDController**: Template for closed-loop control
- **OptimizingController**: Template for MPC

### 3. System Integration

The `ModularThermalSystem` orchestrates component interactions:
```python
for each timestep:
    state = get_system_state()
    control = controller.compute(state)
    update_pumps_and_valves(control)
    update_all_components()
    log_results()
```

## Files

| File | Description |
|------|-------------|
| `components.py` | All physical component classes |
| `controller.py` | Control algorithms |
| `modular_simulation.py` | System integration and main simulation |
| `example_upgrade.py` | Demonstrates component upgrades |
| `ARCHITECTURE.md` | Detailed architecture documentation |

## Running the Simulation

### Basic Simulation
```bash
uv run modular_simulation.py
```

### With Improved Components
```bash
uv run example_upgrade.py
```

## Results

The simulation produces a comprehensive plot showing:

1. **System Temperatures**: Tank, collector, load supply/return vs ambient
2. **Solar Collection**: Irradiance and heat power collected
3. **Load Matching**: Demand vs actual heat delivered
4. **Pump Control**: Solar and load pump speeds over time
5. **Energy Storage**: Tank stored energy (MJ)
6. **Cumulative Energy**: Total solar collected vs load served

### Sample Output
```
Running modular simulation for 48.0 hours...
Initial tank temperature: 30.0°C
Thermal load: variable

Simulation complete!
Final tank temperature: 97.0°C
Max tank temperature: 97.0°C

Energy Summary:
  Solar energy collected: 16.9 MJ
  Load demand: 121.2 MJ
  Load served: 69.8 MJ
  Load fraction met: 57.6%
```

## Component Interfaces

### FluidComponent Base Class
```python
def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update component for one timestep.

    Args:
        dt: Timestep (seconds)
        inputs: Dictionary of input values from connected components

    Returns:
        Dictionary of output values for downstream components
    """
```

### Controller Base Class
```python
def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute control actions based on system state.

    Args:
        system_state: Current state of all components

    Returns:
        Dictionary of control commands (pump speeds, valve positions)
    """
```

## Upgrading Components

### Example: Improved Solar Collector

**Before (Basic Model)**:
```python
Q_absorbed = irradiance * area * absorptance * efficiency
Q_loss = U_L * area * (T_collector - T_ambient)
```

**After (Enhanced Model with IAM)**:
```python
IAM = calculate_incidence_angle_modifier(time)
Q_absorbed = irradiance * area * absorptance * efficiency * IAM
Q_loss_conv = U_L * area * (T_collector - T_ambient)
Q_loss_rad = sigma * area * (T_collector^4 - T_ambient^4)
Q_loss = Q_loss_conv + Q_loss_rad
```

**Integration**: No system code changes needed!
```python
# Just swap the component class
collector = EnhancedSolarCollector("SC1", params)
# Everything else stays the same
```

## System Configuration

### Default Parameters

**Solar Collector**:
- Area: 2.5 m²
- Efficiency: 75%
- Heat loss coefficient: 5.0 W/(m²·K)

**Storage Tank**:
- Volume: 200 liters
- Mass: 200 kg
- Insulation: 1.5 W/(m²·K)

**Thermal Load**:
- Base load: 300 W
- Peak load: 1500 W
- Profile: Variable (peaks morning/evening)

**Control**:
- Solar pump on threshold: ΔT > 7°C
- Solar pump off threshold: ΔT < 3°C
- Tank max temperature: 75°C

## Future Enhancements

Thanks to the modular architecture, these can be added easily:

### New Components
- [ ] Backup heater (electric/gas)
- [ ] Dump load (over-temperature protection)
- [ ] Heat exchanger (indirect systems)
- [ ] Multiple tanks (series/parallel)
- [ ] Expansion tank (pressure control)

### Component Upgrades
- [ ] Stratified storage tank (multi-node)
- [ ] Pump efficiency and power consumption
- [ ] Realistic valve characteristics
- [ ] Building thermal mass model for load
- [ ] Weather-reactive load profiles

### Control Improvements
- [ ] Model Predictive Control (MPC)
- [ ] Weather forecast integration
- [ ] Cost optimization (utility rates)
- [ ] Load prediction
- [ ] Self-tuning PID

### System Features
- [ ] Hydraulic network solver
- [ ] Pressure drop calculations
- [ ] Glycol concentration effects
- [ ] Real weather data integration
- [ ] Economic analysis

## Validation

### Component-Level
Each component should be validated independently:
- Unit tests for energy balance
- Comparison to reference data (ASHRAE, NREL, etc.)
- Parameter sensitivity analysis

### System-Level
- Energy balance verification (< 1% error)
- Temperature range validation
- Comparison to measured system performance
- Control stability analysis

## Comparison to Baseline

| Aspect | Baseline (v1.0) | Modular System |
|--------|----------------|----------------|
| Architecture | Monolithic class | Component-based |
| Thermal load | None | Variable load profiles |
| Control | None | Active pump/valve control |
| Valves | None | Included |
| Upgradability | Difficult | Easy - swap components |
| Extensibility | Limited | High - add components |
| Validation | System-level only | Component + system |

## Dependencies

```toml
[project.dependencies]
numpy >= 1.20.0
matplotlib >= 3.3.0
```

## Quick Start

```python
from components import SolarCollector, StorageTank, ThermalLoad
from controller import BasicController
from modular_simulation import ModularThermalSystem

# Create components
collector = SolarCollector("SC1", SolarCollectorParams(area=3.0))
tank = StorageTank("Tank", StorageTankParams(volume=0.3))
load = ThermalLoad("Load", ThermalLoadParams(load_profile='scheduled'))
controller = BasicController()

# Assemble and run
system = ModularThermalSystem(collector, tank, load, controller)
system.run_simulation(duration_hours=72.0)
system.plot_results()
```

## License & Usage

This code is provided as part of the Passive Logic interview demonstration.
Feel free to use, modify, and extend for your purposes.

## Contact

For questions about the architecture or implementation, please refer to:
- `ARCHITECTURE.md` - Detailed architecture documentation
- `components.py` - Component implementation details
- `controller.py` - Control algorithm details
