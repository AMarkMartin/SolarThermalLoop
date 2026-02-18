"""
Unit tests for system components
Validates energy conservation and physics
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../')

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    ThermalLoad, ThermalLoadParams,
    Pump, Valve
)


def _small_collector_params(**overrides):
    """Create small-scale collector params for unit tests (not the commercial defaults)."""
    defaults = dict(
        area=2.0, efficiency=0.75, absorptance=0.95,
        heat_loss_coef=5.0, thermal_mass=10.0, specific_heat=500.0,
    )
    defaults.update(overrides)
    return SolarCollectorParams(**defaults)


def _small_tank_params(**overrides):
    """Create small-scale tank params for unit tests (not the commercial defaults)."""
    defaults = dict(
        volume=0.2, mass=200.0, specific_heat=4186.0,
        surface_area=1.5, heat_loss_coef=1.5, num_nodes=1,
    )
    defaults.update(overrides)
    return StorageTankParams(**defaults)


class TestSolarCollectorEnergyBalance:
    """Test solar collector energy conservation"""

    def test_energy_conservation_steady_state(self):
        """Energy in = Energy out at steady state"""
        collector = SolarCollector("Test", _small_collector_params())

        # Run for 200 steps to reach steady state
        dt = 60.0
        inputs = {
            'T_inlet': 30.0,
            'flow_rate': 0.03,
            'irradiance': 800.0,
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        for _ in range(200):
            result = collector.update(dt, inputs)

        # At steady state: Q_collected ≈ Q_to_fluid (small difference for thermal mass)
        Q_collected = result['Q_collected']
        Q_to_fluid = result['Q_to_fluid']

        rel_error = abs(Q_collected - Q_to_fluid) / max(abs(Q_collected), 1.0)
        assert rel_error < 0.05, \
            f"Energy balance error {rel_error*100:.1f}% exceeds 5%"

    def test_no_collection_at_night(self):
        """No heat collection when irradiance is zero"""
        collector = SolarCollector("Test", _small_collector_params())

        inputs = {
            'T_inlet': 30.0,
            'flow_rate': 0.03,
            'irradiance': 0.0,  # Night
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        result = collector.update(60.0, inputs)

        # Should have heat loss, no gain
        assert result['Q_collected'] <= 0, "Should not collect heat at night"

    def test_heat_loss_when_hot(self):
        """Collector loses heat when hotter than ambient"""
        collector = SolarCollector("Test", _small_collector_params())
        collector.T_collector = 60.0  # Hot collector

        inputs = {
            'T_inlet': 30.0,
            'flow_rate': 0.0,  # No flow
            'irradiance': 0.0,  # No sun
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        result = collector.update(60.0, inputs)

        # Should cool down
        assert result['Q_collected'] < 0, "Hot collector should lose heat"

    def test_outlet_hotter_than_inlet_when_collecting(self):
        """Outlet temperature should exceed inlet when collecting (after warmup)"""
        collector = SolarCollector("Test", _small_collector_params())

        inputs = {
            'T_inlet': 30.0,
            'flow_rate': 0.03,
            'irradiance': 800.0,
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        # Warm up collector to reach equilibrium
        for _ in range(50):
            result = collector.update(60.0, inputs)

        assert result['T_outlet'] > inputs['T_inlet'], \
            "Outlet should be hotter than inlet when collecting solar"


class TestStorageTankEnergyBalance:
    """Test storage tank energy conservation"""

    def test_energy_conservation(self):
        """Tank energy change equals net heat flow"""
        tank = StorageTank("Test", _small_tank_params(), initial_temp=40.0)

        dt = 60.0
        inputs = {
            'T_inlet_solar': 50.0,
            'flow_rate_solar': 0.03,
            'T_inlet_load': 30.0,
            'flow_rate_load': 0.02,
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        # Store initial energy
        E_initial = tank.params.mass * tank.params.specific_heat * tank.T_tank

        result = tank.update(dt, inputs)

        # Calculate energy change
        E_final = tank.params.mass * tank.params.specific_heat * tank.T_tank
        dE = E_final - E_initial

        # Net heat flow
        Q_net = (result['Q_solar'] + result['Q_load'] - result['Q_loss']) * dt

        # Should match within numerical precision
        rel_error = abs(dE - Q_net) / max(abs(Q_net), 1.0)
        assert rel_error < 0.01, \
            f"Tank energy balance error {rel_error*100:.2f}% exceeds 1%"

    def test_heat_extraction_cools_tank(self):
        """Extracting heat should cool the tank"""
        tank = StorageTank("Test", _small_tank_params(), initial_temp=50.0)

        T_initial = tank.T_tank

        # Extract heat with cool return water
        inputs = {
            'T_inlet_solar': 50.0,  # No solar input
            'flow_rate_solar': 0.0,
            'T_inlet_load': 30.0,  # Cool return
            'flow_rate_load': 0.05,  # High flow
            'T_ambient': 20.0,
            'fluid_cp': 4186.0
        }

        for _ in range(100):  # Run for a while
            result = tank.update(60.0, inputs)

        assert tank.T_tank < T_initial, "Tank should cool when heat is extracted"

    def test_no_temperature_change_without_flow(self):
        """Tank temperature should not change significantly without flow"""
        tank = StorageTank("Test", _small_tank_params(
            heat_loss_coef=0.0  # No losses for this test
        ), initial_temp=40.0)

        T_initial = tank.T_tank

        inputs = {
            'T_inlet_solar': 50.0,
            'flow_rate_solar': 0.0,  # No flow
            'T_inlet_load': 30.0,
            'flow_rate_load': 0.0,  # No flow
            'T_ambient': 20.0,
            'fluid_cp': 3500.0
        }

        tank.update(60.0, inputs)

        assert abs(tank.T_tank - T_initial) < 0.01, \
            "Tank temperature should not change without flow"


class TestThermalLoad:
    """Test thermal load component"""

    def test_constant_load_profile(self):
        """Constant load should return base load"""
        load = ThermalLoad("Test", ThermalLoadParams(
            base_load=500.0,
            load_profile='constant'
        ))

        inputs = {
            'T_inlet': 50.0,
            'flow_rate': 0.02,
            'time_hours': 10.0,
            'fluid_cp': 4186.0
        }

        result = load.update(60.0, inputs)

        assert abs(result['Q_demand'] - 500.0) < 1.0, \
            "Constant load should return base load value"

    def test_variable_load_increases_during_peak(self):
        """Variable load should increase during peak hours"""
        load = ThermalLoad("Test", ThermalLoadParams(
            base_load=500.0,
            peak_load=2000.0,
            load_profile='variable'
        ))

        # Morning peak
        inputs_peak = {
            'T_inlet': 50.0,
            'flow_rate': 0.02,
            'time_hours': 7.0,
            'fluid_cp': 4186.0
        }

        # Off-peak
        inputs_offpeak = {
            'T_inlet': 50.0,
            'flow_rate': 0.02,
            'time_hours': 14.0,
            'fluid_cp': 4186.0
        }

        result_peak = load.update(60.0, inputs_peak)
        result_offpeak = load.update(60.0, inputs_offpeak)

        assert result_peak['Q_demand'] > result_offpeak['Q_demand'], \
            "Demand should be higher during peak hours"

    def test_outlet_cooler_than_inlet(self):
        """Load outlet should be cooler than inlet (heat extraction)"""
        load = ThermalLoad("Test", ThermalLoadParams(base_load=1000.0))

        inputs = {
            'T_inlet': 50.0,
            'flow_rate': 0.03,
            'time_hours': 10.0,
            'fluid_cp': 4186.0
        }

        result = load.update(60.0, inputs)

        assert result['T_outlet'] < inputs['T_inlet'], \
            "Outlet should be cooler than inlet after heat extraction"


class TestPumpAndValve:
    """Test pump and valve components"""

    def test_pump_zero_flow_at_zero_speed(self):
        """Pump should have zero flow at zero speed"""
        pump = Pump("Test", max_flow=0.1)

        result = pump.update(60.0, {'speed': 0.0})

        assert result['flow_rate'] == 0.0, "Flow should be zero at zero speed"

    def test_pump_max_flow_at_full_speed(self):
        """Pump should approach max flow at full speed"""
        pump = Pump("Test", max_flow=0.1)

        result = pump.update(60.0, {'speed': 1.0})

        assert abs(result['flow_rate'] - 0.1) < 0.01, \
            "Flow should be near max at full speed"

    def test_valve_zero_flow_when_closed(self):
        """Valve should have zero flow when closed"""
        valve = Valve("Test", max_flow=0.1)

        result = valve.update(60.0, {'position': 0.0})

        assert result['flow_rate'] == 0.0, "Flow should be zero when valve closed"

    def test_valve_max_flow_when_open(self):
        """Valve should have max flow when fully open"""
        valve = Valve("Test", max_flow=0.1)

        result = valve.update(60.0, {'position': 1.0})

        assert abs(result['flow_rate'] - 0.1) < 0.01, \
            "Flow should be near max when valve fully open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
