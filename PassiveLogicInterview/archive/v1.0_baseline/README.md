# Solar Thermal System Simulation

A thermodynamically accurate simulation of heat transfer from a solar panel to a storage tank via a pumped fluid loop.

## System Overview

The simulation models:
- **Solar Panel**: Absorbs solar radiation and heats the circulating fluid
- **Pump**: Circulates heat transfer fluid through the system
- **Storage Tank**: Stores thermal energy in water
- **Environment**: Provides solar input and causes heat losses

## Thermodynamic Principles

### 1. Energy Conservation
All energy transfers follow the first law of thermodynamics:
```
dE/dt = Q_in - Q_out - Q_loss
```

### 2. Solar Panel Heat Transfer
The panel absorbs solar radiation with losses due to convection and radiation:
```
Q_solar = (I × A × α × η) - (U_L × A × (T_panel - T_ambient))
```
Where:
- I = solar irradiance (W/m²)
- A = panel area (m²)
- α = absorptance (0.95)
- η = collector efficiency (0.75)
- U_L = heat loss coefficient (W/m²·K)

### 3. Fluid Energy Transport
Heat carried by flowing fluid:
```
Q = ṁ × cp × (T_out - T_in)
```
Where:
- ṁ = mass flow rate (kg/s)
- cp = specific heat capacity (J/kg·K)
- T = temperature (K or °C)

### 4. Storage Tank Energy Balance
Tank temperature changes based on net heat transfer:
```
m_tank × cp_tank × dT/dt = Q_in - Q_loss
```

Heat loss to ambient:
```
Q_loss = U × A_tank × (T_tank - T_ambient)
```

## Key Features

✓ **Realistic Solar Model**: Sinusoidal irradiance profile simulating daylight hours
✓ **Temperature-Dependent Losses**: Heat loss increases with temperature differential
✓ **Fluid Properties**: Uses water-glycol mixture properties (50% glycol)
✓ **Time-Varying Simulation**: 24-hour simulation with 1-minute time steps
✓ **Energy Balance Validation**: Tracks cumulative energy flows

## Installation

Requires Python 3.7+ with:
```bash
pip install numpy matplotlib
```

## Usage

Run the simulation:
```bash
python solar_thermal_simulation.py
```

This will:
1. Simulate 24 hours of system operation
2. Generate plots showing temperatures, heat flows, and energy balance
3. Save results as `solar_thermal_results.png`
4. Print thermodynamic analysis to console

## System Parameters

### Solar Panel
- Area: 2.0 m²
- Efficiency: 75%
- Absorptance: 95%
- Heat loss coefficient: 5.0 W/(m²·K)

### Storage Tank
- Volume: 150 liters
- Mass: 150 kg (water)
- Specific heat: 4186 J/(kg·K)
- Heat loss coefficient: 1.5 W/(m²·K) (insulated)

### Heat Transfer Fluid
- Density: 1040 kg/m³ (50% glycol)
- Specific heat: 3500 J/(kg·K)
- Flow rate: 0.02 kg/s (72 kg/hr)

### Environment
- Ambient temperature: 20°C
- Peak solar irradiance: 1000 W/m²
- Solar hours: 6:00 - 18:00

## Results Interpretation

The simulation produces four plots:

1. **System Temperatures**: Shows tank and panel outlet temperatures over time
2. **Solar Input and Heat Collection**: Irradiance and actual heat collected
3. **Net Heat Transfer to Tank**: Heat flow accounting for losses
4. **Cumulative Energy Balance**: Total energy collected vs stored

### Expected Behavior

- Tank temperature rises during daylight hours (6:00-18:00)
- Peak temperatures occur in the afternoon
- Tank cools slowly overnight due to insulation
- System efficiency typically 50-70% of theoretical maximum due to losses

## Thermodynamic Validation

The simulation ensures:
- Energy is conserved at every time step
- Heat flows from hot to cold (second law)
- No negative absolute temperatures
- Realistic efficiency based on operating conditions
- Heat losses proportional to temperature differentials

## Customization

Modify parameters in `main()` to explore different scenarios:
- Change panel area or efficiency
- Adjust tank size
- Vary flow rate
- Modify ambient temperature
- Test different fluid properties

## Physical Accuracy

The model includes:
- ✓ Transient (time-dependent) behavior
- ✓ Heat capacity effects
- ✓ Environmental heat losses
- ✓ Flow rate effects on heat transfer
- ✓ Temperature-dependent performance

Simplifications:
- Uniform tank temperature (perfectly mixed)
- Constant fluid properties
- Simplified solar model (actual irradiance varies with cloud cover, angle)
- Ideal heat exchanger assumption in tank
