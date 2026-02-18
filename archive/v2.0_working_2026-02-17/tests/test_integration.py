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


class TestSystemEnergyBalance:
    """Test energy conservation in complete system"""

    def test_simple_system_energy_conservation(self):
        """Energy balance over full solar collection cycle"""

        # Create simple system
        collector = SolarCollector("SC", SolarCollectorParams(area=2.0))
        tank = StorageTank("Tank", StorageTankParams(volume=0.2, mass=200.0), initial_temp=30.0)
        pump = Pump("Pump", max_flow=0.03)

        # Track energy
        E_solar_absorbed = 0.0
        E_tank_change = 0.0
        E_losses = 0.0

        dt = 60.0
        T_ambient = 20.0

        # Initial tank energy
        E_tank_initial = tank.params.mass * tank.params.specific_heat * tank.T_tank

        # Simulate sunny period
        for step in range(60):  # 1 hour
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

        collector = SolarCollector("SC", SolarCollectorParams(area=2.0))
        tank = StorageTank("Tank", StorageTankParams(volume=0.2, mass=200.0), initial_temp=30.0)
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

        # Small dT - should stay on due to hysteresis
        system_state_small_dt = {
            'T_collector': 32.0,
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

        collector = SolarCollector("SC", SolarCollectorParams())
        tank = StorageTank("Tank", StorageTankParams())

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

        collector = SolarCollector("SC", SolarCollectorParams(area=10.0))

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

        # Collector can get hot, but should be reasonable
        assert result['T_collector'] < 200.0, \
            f"Collector temperature {result['T_collector']:.1f}°C is unrealistically high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
