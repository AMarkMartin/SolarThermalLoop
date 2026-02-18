# Unit Tests

## Overview

Comprehensive unit tests for the solar thermal system simulation to verify correct physics and energy conservation.

## Test Modules

### 1. `test_solar_radiation.py`
Tests the solar radiation model for accuracy.

**Tests**:
- ✅ Solar position calculations (altitude, azimuth)
- ✅ Declination angle at solstices and equinoxes
- ✅ Clear sky radiation model
- ✅ Cloud cover effects on direct/diffuse components
- ✅ Tilted surface calculations
- ✅ Morning/afternoon symmetry

**Key Validations**:
- Summer solstice declination: ~23.45°
- Winter solstice declination: ~-23.45°
- Solar noon altitude matches theory
- Irradiance 900-1200 W/m² for clear sky at high altitude

### 2. `test_components.py`
Tests individual component energy balance and physics.

**Tests**:
- ✅ Solar collector energy conservation
- ✅ Storage tank energy balance
- ✅ Thermal load behavior
- ✅ Pump and valve flow control
- ✅ Temperature limits

**Key Validations**:
- Energy in = Energy out (within 1-5%)
- Collector outlet hotter than inlet when collecting
- Tank temperature changes match heat flows
- No energy creation or destruction

### 3. `test_pumps.py`
Tests pump performance curves and efficiency.

**Tests**:
- ✅ Pump curves (head vs flow)
- ✅ Efficiency curves with BEP
- ✅ Power consumption calculations
- ✅ Affinity laws (flow scales with speed)
- ✅ Energy accumulation over time

**Key Validations**:
- Shutoff head matches specification
- Head decreases with increasing flow
- Efficiency peaks at BEP
- Power consumption realistic (P = ρ·g·Q·H/η)

### 4. `test_integration.py`
Tests complete system behavior and integration.

**Tests**:
- ✅ System-level energy conservation
- ✅ Controller logic (differential temp control)
- ✅ Controller hysteresis
- ✅ Over-temperature protection
- ✅ Temperature limit enforcement
- ✅ Multi-component interactions

**Key Validations**:
- System energy balance < 5% error
- Tank heats up when collecting solar
- Controller turns pump on/off appropriately
- No negative or super-boiling temperatures

## Running Tests

### Install Dependencies
```bash
cd tests
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest
```

### Run Specific Test Module
```bash
pytest test_solar_radiation.py -v
```

### Run with Coverage
```bash
pytest --cov=../src --cov-report=html
```

### Run Specific Test Class
```bash
pytest test_components.py::TestSolarCollectorEnergyBalance -v
```

### Run Specific Test
```bash
pytest test_solar_radiation.py::TestSolarPosition::test_solar_declination_summer_solstice -v
```

## Test Coverage Goals

- **Component Energy Balance**: 100% (critical for physics)
- **Solar Radiation**: >95% (high confidence needed)
- **Controllers**: >90% (logic coverage)
- **Integration**: >85% (system behavior)

## Expected Test Results

All tests should pass with these characteristics:

### Solar Radiation Tests
- **test_solar_declination_summer_solstice**: ±1° tolerance
- **test_solar_altitude_solar_noon**: ±3° tolerance
- **test_clear_sky_radiation**: 900-1200 W/m² range

### Component Tests
- **Energy balance**: <1% error for tank, <5% for transient components
- **Temperature behavior**: Physical limits enforced

### Pump Tests
- **Efficiency**: Peaks at BEP as expected
- **Power consumption**: Within 10% of theoretical

### Integration Tests
- **System energy balance**: <5% error over full cycle
- **Controller behavior**: Proper on/off logic with hysteresis

## Adding New Tests

When adding new components or features:

1. **Create corresponding test file** or add to existing
2. **Test energy conservation** if component handles energy
3. **Test edge cases**: Zero flow, extreme temperatures, etc.
4. **Test integration**: How does it affect system behavior?

### Template for New Component Test

```python
def test_component_energy_conservation(self):
    """Component should conserve energy"""
    component = NewComponent(...)

    # Initial state
    E_initial = calculate_energy(component)

    # Run update
    inputs = {...}
    result = component.update(dt, inputs)

    # Final state
    E_final = calculate_energy(component)

    # Energy balance
    dE = E_final - E_initial
    Q_net = (result['Q_in'] - result['Q_out']) * dt

    assert abs(dE - Q_net) / max(abs(Q_net), 1.0) < 0.01
```

## Continuous Integration

These tests should be run:
- Before every commit (pre-commit hook)
- On every pull request
- Before releases

## Known Tolerances

| Test Category | Tolerance | Reason |
|--------------|-----------|---------|
| Energy Balance | 1-5% | Numerical precision, thermal mass effects |
| Temperature Calculations | ±1-3°C | Simplified models |
| Solar Position | ±3° | Atmospheric effects not modeled |
| Flow/Pressure | ±10% | Simplified hydraulics |

## Debugging Failed Tests

If tests fail:

1. **Check tolerance**: Is the error within expected numerical precision?
2. **Verify inputs**: Are test inputs realistic?
3. **Check units**: Temperature (°C vs K), pressure (Pa vs bar), etc.
4. **Review physics**: Does the component physics match reality?
5. **Examine trends**: Even if absolute values are off, trends should be correct

## Future Test Additions

- [ ] Stratified tank model validation
- [ ] IAM (Incidence Angle Modifier) tests
- [ ] Real weather data comparison
- [ ] Long-term simulation stability (weeks/months)
- [ ] Glycol property temperature dependence
- [ ] Freeze protection logic
- [ ] Performance degradation over time
