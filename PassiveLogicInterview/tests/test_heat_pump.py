"""
Unit tests for heat pump component and unglazed collector behavior.
Validates COP calculations, energy balance, and SAHP-specific physics.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../')

from src.components import (
    HeatPump, HeatPumpParams,
    SolarCollector, SolarCollectorParams,
    WaterGlycolMixture,
)
from src.control import SAHPController


# ============================================================================
# HEAT PUMP TESTS
# ============================================================================

class TestHeatPumpCOP:
    """Test COP calculations"""

    def test_cop_increases_with_higher_source_temp(self):
        """Higher source temperature should give higher COP"""
        hp = HeatPump("HP", HeatPumpParams())

        result_cold = hp.update(60.0, {'T_tank': 0.0, 'Q_demand': 100000.0})
        cop_cold = result_cold['COP']

        result_warm = hp.update(60.0, {'T_tank': 25.0, 'Q_demand': 100000.0})
        cop_warm = result_warm['COP']

        assert cop_warm > cop_cold, \
            f"COP at 25°C ({cop_warm:.2f}) should exceed COP at 0°C ({cop_cold:.2f})"

    def test_cop_within_realistic_bounds(self):
        """COP should be between 1.5 and 8.0 for typical conditions"""
        hp = HeatPump("HP", HeatPumpParams())

        for T_source in [-5.0, 0.0, 10.0, 20.0, 30.0]:
            result = hp.update(60.0, {'T_tank': T_source, 'Q_demand': 100000.0})
            assert 1.5 <= result['COP'] <= 8.0, \
                f"COP {result['COP']:.2f} at T_source={T_source}°C out of bounds"

    def test_cop_matches_carnot_fraction(self):
        """COP should equal carnot_efficiency * COP_carnot"""
        params = HeatPumpParams(carnot_efficiency=0.45, T_supply=45.0)
        hp = HeatPump("HP", params)

        T_source = 10.0
        T_hot_K = 45.0 + 273.15
        T_cold_K = 10.0 + 273.15
        expected_cop = 0.45 * T_hot_K / (T_hot_K - T_cold_K)

        result = hp.update(60.0, {'T_tank': T_source, 'Q_demand': 100000.0})

        assert abs(result['COP'] - expected_cop) < 0.01, \
            f"COP {result['COP']:.3f} should match expected {expected_cop:.3f}"


class TestHeatPumpEnergyBalance:
    """Test heat pump energy conservation"""

    def test_first_law_energy_balance(self):
        """Q_heating = Q_evaporator + W_compressor (first law)"""
        hp = HeatPump("HP", HeatPumpParams())

        result = hp.update(60.0, {'T_tank': 15.0, 'Q_demand': 200000.0})

        Q_heating = result['Q_heating']
        Q_evap = result['Q_evaporator']
        W_comp = result['W_compressor']

        # First law: Q_heating = Q_evap + W_comp
        balance_error = abs(Q_heating - (Q_evap + W_comp))
        rel_error = balance_error / Q_heating if Q_heating > 0 else 0

        assert rel_error < 0.001, \
            f"Energy balance error {rel_error*100:.3f}% exceeds 0.1%"

    def test_heating_limited_to_capacity(self):
        """Q_heating should not exceed rated capacity"""
        params = HeatPumpParams(heating_capacity=300000.0)
        hp = HeatPump("HP", params)

        result = hp.update(60.0, {'T_tank': 20.0, 'Q_demand': 500000.0})

        assert result['Q_heating'] <= params.heating_capacity, \
            "Heating output should not exceed rated capacity"

    def test_no_output_below_min_source_temp(self):
        """Heat pump should not operate below minimum source temperature"""
        params = HeatPumpParams(min_source_temp=-10.0)
        hp = HeatPump("HP", params)

        result = hp.update(60.0, {'T_tank': -15.0, 'Q_demand': 100000.0})

        assert result['Q_heating'] == 0.0, "Should not heat below min source temp"
        assert result['W_compressor'] == 0.0, "Should not consume power below min source temp"

    def test_no_output_without_demand(self):
        """Heat pump should not operate without heating demand"""
        hp = HeatPump("HP", HeatPumpParams())

        result = hp.update(60.0, {'T_tank': 20.0, 'Q_demand': 0.0})

        assert result['Q_heating'] == 0.0, "Should not heat without demand"

    def test_tank_extraction_equals_evaporator_load(self):
        """Q_extracted_from_tank should equal Q_evaporator"""
        hp = HeatPump("HP", HeatPumpParams())

        result = hp.update(60.0, {'T_tank': 20.0, 'Q_demand': 150000.0})

        assert result['Q_extracted_from_tank'] == result['Q_evaporator'], \
            "Tank extraction should equal evaporator load"


# ============================================================================
# UNGLAZED COLLECTOR TESTS
# ============================================================================

class TestUnglazedCollector:
    """Test unglazed collector specific behavior"""

    def test_higher_losses_than_glazed(self):
        """Unglazed collector should lose more heat than glazed under same conditions"""
        unglazed = SolarCollector("Unglazed", SolarCollectorParams(
            area=2.0, heat_loss_coef=15.0, thermal_mass=10.0, specific_heat=500.0,
        ))
        glazed = SolarCollector("Glazed", SolarCollectorParams(
            area=2.0, heat_loss_coef=5.0, thermal_mass=10.0, specific_heat=500.0,
        ))

        # Set both to same temperature
        unglazed.T_collector = 40.0
        glazed.T_collector = 40.0

        inputs = {
            'T_inlet': 30.0,
            'flow_rate': 0.0,  # No flow — just look at losses
            'irradiance': 0.0,  # No sun
            'T_ambient': 20.0,
            'fluid_cp': 3700.0
        }

        result_unglazed = unglazed.update(60.0, inputs)
        result_glazed = glazed.update(60.0, inputs)

        # Unglazed should lose more heat (more negative Q_collected)
        assert result_unglazed['Q_collected'] < result_glazed['Q_collected'], \
            "Unglazed collector should have higher heat losses"

    def test_useful_collection_at_moderate_irradiance(self):
        """Unglazed collector should still collect useful heat at moderate irradiance"""
        collector = SolarCollector("Unglazed", SolarCollectorParams(
            area=10.0, efficiency=0.90, absorptance=0.92,
            heat_loss_coef=15.0, thermal_mass=50.0, specific_heat=800.0,
        ))

        inputs = {
            'T_inlet': 15.0,
            'flow_rate': 0.1,  # ~0.01 kg/(s·m²), typical for solar loop
            'irradiance': 500.0,  # Moderate sun
            'T_ambient': 10.0,
            'fluid_cp': 3700.0
        }

        # Run to reach equilibrium
        for _ in range(50):
            result = collector.update(60.0, inputs)

        assert result['Q_to_fluid'] > 0, \
            "Unglazed collector should deliver useful heat at moderate irradiance"

    def test_ambient_heat_gain_when_cold(self):
        """
        When collector is below ambient, heat 'loss' is negative
        (net heat gain from convective exchange with ambient air).
        This is a feature of unglazed collectors used in SAHP systems.
        """
        collector = SolarCollector("Unglazed", SolarCollectorParams(
            area=10.0, heat_loss_coef=15.0, thermal_mass=50.0, specific_heat=800.0,
        ))
        collector.T_collector = 5.0  # Below ambient

        inputs = {
            'T_inlet': 5.0,
            'flow_rate': 0.0,
            'irradiance': 0.0,  # Night
            'T_ambient': 15.0,  # Warmer than collector
            'fluid_cp': 3700.0
        }

        result = collector.update(60.0, inputs)

        # Q_collected = Q_absorbed(0) - Q_loss, and Q_loss is negative
        # when T_collector < T_ambient, so Q_collected should be positive
        assert result['Q_collected'] > 0, \
            "Collector below ambient should gain heat from environment"


# ============================================================================
# WATER-GLYCOL FLUID TESTS
# ============================================================================

class TestWaterGlycolMixture:
    """Test fluid property model"""

    def test_default_properties(self):
        """Default 30% glycol should have reasonable properties"""
        fluid = WaterGlycolMixture()

        assert fluid.glycol_fraction == 0.30
        assert 3600 < fluid.specific_heat < 3900
        assert fluid.freeze_point < 0.0

    def test_cp_temperature_dependence(self):
        """cp should increase mildly with temperature"""
        fluid = WaterGlycolMixture()

        cp_cold = fluid.cp_at_temp(0.0)
        cp_hot = fluid.cp_at_temp(60.0)

        assert cp_hot > cp_cold, "cp should increase with temperature"
        # Should be a mild effect — not more than ~5% over 60°C range
        assert (cp_hot - cp_cold) / cp_cold < 0.05, \
            "cp temperature dependence should be mild"


# ============================================================================
# SAHP CONTROLLER TESTS
# ============================================================================

class TestSAHPController:
    """Test SAHP controller logic"""

    def test_solar_pump_on_with_irradiance_and_dt(self):
        """Solar pump should turn on when collector is hotter than tank and sun is up"""
        controller = SAHPController(solar_dT_threshold=5.0)

        state = {
            'T_collector': 40.0,
            'T_tank': 20.0,
            'Q_demand': 0.0,
            'irradiance': 600.0,
        }
        control = controller.compute_control(state)

        assert control['solar_pump_speed'] > 0, "Solar pump should run"

    def test_solar_pump_off_at_night(self):
        """Solar pump should be off at night"""
        controller = SAHPController()

        state = {
            'T_collector': 20.0,
            'T_tank': 25.0,
            'Q_demand': 50000.0,
            'irradiance': 0.0,
        }
        control = controller.compute_control(state)

        assert control['solar_pump_speed'] == 0.0, "Solar pump should be off at night"

    def test_hp_enables_on_demand(self):
        """Heat pump should enable when there is heating demand"""
        controller = SAHPController()

        state = {
            'T_collector': 20.0,
            'T_tank': 15.0,
            'Q_demand': 100000.0,
            'irradiance': 0.0,
        }
        control = controller.compute_control(state)

        assert control['hp_enable'] is True, "HP should enable with demand"

    def test_hp_disabled_without_demand(self):
        """Heat pump should be disabled when no demand"""
        controller = SAHPController()

        state = {
            'T_collector': 30.0,
            'T_tank': 25.0,
            'Q_demand': 0.0,
            'irradiance': 500.0,
        }
        control = controller.compute_control(state)

        assert control['hp_enable'] is False, "HP should be off without demand"

    def test_lower_tank_max_temp(self):
        """SAHP controller should stop solar pump at lower tank temp than BasicController"""
        controller = SAHPController(tank_max_temp=45.0)

        # Tank near SAHP max — solar pump should stop
        state = {
            'T_collector': 55.0,
            'T_tank': 46.0,
            'Q_demand': 0.0,
            'irradiance': 800.0,
        }
        control = controller.compute_control(state)

        assert control['solar_pump_speed'] == 0.0, \
            "Solar pump should stop when tank exceeds SAHP max temp (45°C)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
