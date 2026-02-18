# Solar Thermal System Simulation

**Version 2.0** - Modular architecture with realistic physics models

Thermodynamically accurate simulation of solar thermal heating systems with modular, upgradeable components.

---

## Quick Start

### Run Winter Heating Scenario
```bash
uv run examples/winter_scenario.py
```

### Run Tests
```bash
cd tests
pytest -v
```

---

## Project Structure

```
PassiveLogicInterview/
├── src/                       # Source code
│   ├── components.py          # Component base classes & implementations
│   ├── models.py              # Physics models (solar, pumps, building)
│   └── control.py             # Control algorithms
│
├── examples/                  # Example simulations
│   ├── winter_scenario.py     # Winter heating (recommended)
│   ├── modular_simulation.py  # Modular system demo
│   └── solar_thermal_simulation.py  # Baseline version
│
├── tests/                     # Unit tests
│   ├── test_solar_radiation.py    # Sun position, irradiance
│   ├── test_components.py         # Energy balance, physics
│   ├── test_pumps.py              # Pump curves, efficiency
│   └── test_integration.py        # System-level validation
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md
│   ├── WINTER_SCENARIO_SUMMARY.md
│   └── PROJECT_STATUS.md
│
├── results/                   # Output plots
├── archive/                   # Version history
└── pyproject.toml
```

---

## Features

### ✅ Realistic Physics Models

| Component | Fidelity | Status |
|-----------|----------|--------|
| **Solar Radiation** | High | Production ready ✅ |
| **Pump Curves** | High | Production ready ✅ |
| **Solar Collector** | Medium | Good, can add IAM |
| **Storage Tank** | Medium | Fully-mixed model |
| **Building Load** | Medium | Simplified but adequate |

### ✅ Modular Architecture

- **Independent components**: Upgrade physics without affecting integration
- **Well-defined interfaces**: Each component has clear input/output contracts
- **Validated components**: Unit tests verify energy conservation

### ✅ Advanced Features

- Geographic solar position calculations (latitude, season, time)
- Weather forecasting capability
- Pump performance curves with efficiency tracking
- Building thermal mass (simplified)
- Control system with hysteresis
- Economic analysis (fuel costs, savings)

---

## Current Capabilities

### Winter Heating Scenario (Primary Demo)

**System Configuration**:
- 4 m² solar collector (supplemental heating)
- 300L storage tank
- 1000 m² building
- 15 kW design heat load

**Results** (5-day simulation):
- **Solar Fraction**: 12.5% average
- **Building Demand**: 635.7 kWh
- **Solar Delivered**: 92.0 kWh
- **Auxiliary Heating**: 543.6 kWh (natural gas)
- **Operating Cost**: $21.77 (5 days)
- **Solar Savings**: $3.68

**Physics Validated**:
- ✅ Energy conservation (< 5% error)
- ✅ Demand correlates with weather
- ✅ Tank temperatures realistic (15-40°C)
- ✅ Pump parasitic power < 1%

---

## Unit Tests

### Test Coverage

**4 test modules, 40+ tests**:

1. **test_solar_radiation.py**: Sun position, irradiance components
2. **test_components.py**: Energy balance, temperature limits
3. **test_pumps.py**: Performance curves, efficiency, power
4. **test_integration.py**: System energy balance, controller logic

### Run Tests
```bash
cd tests
pytest                    # Run all
pytest -v                 # Verbose
pytest --cov=../src       # With coverage
```

**Expected**: All tests pass with realistic tolerances

---

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and modularity |
| [WINTER_SCENARIO_SUMMARY.md](docs/WINTER_SCENARIO_SUMMARY.md) | Winter simulation analysis |
| [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Current status and issues |
| [tests/README.md](tests/README.md) | Test documentation |

---

## Development

### Dependencies
```bash
# Runtime
numpy >= 1.20.0
matplotlib >= 3.3.0

# Testing
pytest >= 7.0.0
pytest-cov >= 4.0.0
```

### Install
```bash
uv sync
```

---

## Version History

- **v2.0** - Winter scenario, modular architecture, comprehensive tests
- **v1.0** - Baseline simulation (archived)

See `archive/` for version history.

---

**Status**: ✅ Ready for solar loop physics refinement
