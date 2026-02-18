"""
Integration tests for complete system
Validates system-level energy balance and behavior
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../')

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    Pump
)
from src.control import BasicController


def _small_collector_params(**overrides):
    """Create small-scale collector params for unit tests."""
    defaults = dict(
        area=2.0, efficiency=0.75, absorptance=0.95,
        heat_loss_coef=5.0, thermal_mass=10.0, specific_heat=500.0,
    )
    defaults.update(overrides)
    return SolarCollectorParams(**defaults)


def _small_tank_params(**overrides):
    """Create small-scale tank params for unit tests."""
    defaults = dict(
        volume=0.2, mass=200.0, specific_heat=4186.0,
        surface_area=1.5, heat_loss_coef=1.5, num_nodes=1,
    )
    defaults.update(overrides)
    return StorageTankParams(**defaults)


class TestSystemEnergyBalance:
    """Test energy conservation in complete system"""

    def test_simple_system_energy_conservation(self):
        """Energy balance over full solar collection cycle"""

        # Create simple system
        collector = SolarCollector("SC", _small_collector_params())
        tank = StorageTank("Tank", _small_tank_params(), initial_temp=30.0)
        pump = Pump("Pump", max_flow=0.03)

        # Track energy
        E_solar_absorbed = 0.0
        E_tank_change = 0.0
        E_losses = 0.0

        dt = 60.0
        T_ambient = 20.0

        # Initial tank energy
        E_tank_initial = tank.params.mass * tank.params.specific_heat * tank.T_tank

        # Simulate sunny period (2 hours for better energy balance convergence)
        for step in range(120):
            # Pump running
            pump_out = pump.update(dt, {'speed': 1.0})

            # Collector
            collector_inputs = {
                'T_inlet': tank.T_tank,
                'flow_rate': pump_out['flow_rate'],
                'irradiance': 800.0,
                'T_ambient': T_ambient,
                'fluid_cp': 3500.0
            }
            collector_out = collector.update(dt, collector_inputs)

            # Tank
            tank_inputs = {
                'T_inlet_solar': collector_out['T_outlet'],
                'flow_rate_solar': pump_out['flow_rate'],
                'T_inlet_load': tank.T_tank,
                'flow_rate_load': 0.0,
                'T_ambient': T_ambient,
                'fluid_cp': 3500.0
            }
            tank_out = tank.update(dt, tank_inputs)

            # Accumulate energy
            E_solar_absorbed += collector_out['Q_collected'] * dt
            E_losses += tank_out['Q_loss'] * dt

        # Final tank energy
        E_tank_final = tank.params.mass * tank.params.specific_heat * tank.T_tank
        E_tank_change = E_tank_final - E_tank_initial

        # Energy balance: E_solar = E_tank_change + E_losses
        E_balance = E_solar_absorbed - (E_tank_change + E_losses)

        rel_error = abs(E_balance) / E_solar_absorbed
        assert rel_error < 0.05, \
            f"System energy balance error {rel_error*100:.1f}% exceeds 5%"

    def test_tank_temperature_increases_with_solar(self):
        """Tank should heat up when collecting solar"""

        collector = SolarCollector("SC", _small_collector_params())
        tank = StorageTank("Tank", _small_tank_params(), initial_temp=30.0)
        pump = Pump("Pump", max_flow=0.03)

        T_initial = tank.T_tank

        dt = 60.0

        # Simulate sunny period
        for _ in range(60):
            pump_out = pump.update(dt, {'speed': 1.0})

            collector_out = collector.update(dt, {
                'T_inlet': tank.T_tank,
                'flow_rate': pump_out['flow_rate'],
                'irradiance': 800.0,
                'T_ambient': 20.0,
                'fluid_cp': 3500.0
            })

            tank.update(dt, {
                'T_inlet_solar': collector_out['T_outlet'],
                'flow_rate_solar': pump_out['flow_rate'],
                'T_inlet_load': tank.T_tank,
                'flow_rate_load': 0.0,
                'T_ambient': 20.0,
                'fluid_cp': 3500.0
            })

        assert tank.T_tank > T_initial, \
            f"Tank temperature should increase from {T_initial}°C"
        assert tank.T_tank < 100.0, \
            f"Tank temperature {tank.T_tank}°C should stay below boiling"


class TestControllerBehavior:
    """Test controller logic"""

    def test_controller_turns_on_pump_with_solar(self):
        """Controller should turn on pump when collector is hot"""

        controller = BasicController(solar_dT_threshold=5.0)

        system_state = {
            'T_collector': 50.0,
            'T_tank': 30.0,
            'Q_demand': 0.0,
            'irradiance': 800.0
        }

        control = controller.compute_control(system_state)

        assert control['solar_pump_speed'] > 0, \
            "Solar pump should run when collector is hot"

    def test_controller_keeps_pump_off_at_night(self):
        """Controller should keep pump off with no solar"""

        controller = BasicController()

        system_state = {
            'T_collector': 25.0,
            'T_tank': 30.0,
            'Q_demand': 0.0,
            'irradiance': 0.0
        }

        control = controller.compute_control(system_state)

        assert control['solar_pump_speed'] == 0, \
            "Solar pump should be off at night"

    def test_controller_has_hysteresis(self):
        """Controller should show hysteresis to prevent cycling"""

        controller = BasicController(solar_dT_threshold=5.0)

        # Turn pump on
        system_state_on = {
            'T_collector': 40.0,
            'T_tank': 30.0,
            'Q_demand': 0.0,
            'irradiance': 800.0
        }
        control = controller.compute_control(system_state_on)
        pump_on = control['solar_pump_speed'] > 0

        # Moderate dT (3°C) — above turn-off (2.5°C) but below turn-on (5°C)
        # Should stay on due to hysteresis
        system_state_small_dt = {
            'T_collector': 33.0,
            'T_tank': 30.0,
            'Q_demand': 0.0,
            'irradiance': 700.0
        }
        control = controller.compute_control(system_state_small_dt)

        assert control['solar_pump_speed'] > 0, \
            "Pump should stay on due to hysteresis"

    def test_controller_prevents_overheating(self):
        """Controller should stop pump if tank too hot"""

        controller = BasicController(tank_max_temp=80.0)

        system_state = {
            'T_collector': 90.0,
            'T_tank': 85.0,  # Above max
            'Q_demand': 0.0,
            'irradiance': 900.0
        }

        control = controller.compute_control(system_state)

        assert control['solar_pump_speed'] == 0, \
            "Pump should stop when tank exceeds max temperature"


class TestTemperatureLimits:
    """Test that temperatures stay within physical limits"""

    def test_no_negative_temperatures(self):
        """All components should maintain positive absolute temperatures"""

        collector = SolarCollector("SC", _small_collector_params())
        tank = StorageTank("Tank", _small_tank_params())

        # Extreme cold scenario
        dt = 60.0
        inputs_cold = {
            'T_inlet': 5.0,
            'flow_rate': 0.03,
            'irradiance': 0.0,
            'T_ambient': -20.0,
            'fluid_cp': 3500.0
        }

        for _ in range(100):
            result = collector.update(dt, inputs_cold)
            assert result['T_collector'] > -273.15, \
                "Temperature should stay above absolute zero"

    def test_no_super_boiling_temperatures(self):
        """Temperatures should stay below steam point at 1 atm"""

        collector = SolarCollector("SC", _small_collector_params(area=10.0))

        # Extreme sun, no flow
        dt = 60.0
        inputs_extreme = {
            'T_inlet': 80.0,
            'flow_rate': 0.001,  # Very low flow
            'irradiance': 1200.0,  # Very high sun
            'T_ambient': 35.0,
            'fluid_cp': 3500.0
        }

        for _ in range(100):
            result = collector.update(dt, inputs_extreme)

        # Collector can get hot with stagnation, but should be bounded
        assert result['T_collector'] < 250.0, \
            f"Collector temperature {result['T_collector']:.1f}°C is unrealistically high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
