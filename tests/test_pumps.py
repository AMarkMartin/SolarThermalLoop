"""
Unit tests for pump performance curves
Validates pump physics and efficiency
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../')

from src.models import PumpWithCurve


class TestPumpCurves:
    """Test pump performance curve calculations"""

    def test_shutoff_head_at_zero_flow(self):
        """Pump should produce max head at zero flow"""
        pump = PumpWithCurve(
            "Test",
            rated_flow=0.05,
            rated_head=5.0,
            max_head=7.5
        )

        # Zero speed first
        pump.speed = 0.0
        head = pump.pump_curve(flow_fraction=0.0)
        assert head == 0.0, "Head should be zero at zero speed"

        # Now at full speed with zero flow
        pump.speed = 1.0
        head = pump.pump_curve(flow_fraction=0.0)

        assert abs(head - 7.5) < 0.1, \
            f"Shutoff head {head}m should equal max_head 7.5m"

    def test_head_decreases_with_flow(self):
        """Head should decrease as flow increases"""
        pump = PumpWithCurve(
            "Test",
            rated_flow=0.05,
            rated_head=5.0,
            max_head=7.5
        )
        pump.speed = 1.0

        head_low_flow = pump.pump_curve(flow_fraction=0.2)
        head_high_flow = pump.pump_curve(flow_fraction=0.8)

        assert head_low_flow > head_high_flow, \
            "Head should decrease with increasing flow"

    def test_efficiency_peaks_at_bep(self):
        """Efficiency should peak near best efficiency point"""
        pump = PumpWithCurve(
            "Test",
            rated_flow=0.05,
            efficiency_bep=0.70
        )

        # Efficiency at various flow fractions
        eta_low = pump.efficiency_curve(flow_fraction=0.3)
        eta_bep = pump.efficiency_curve(flow_fraction=1.0)
        eta_high = pump.efficiency_curve(flow_fraction=1.5)

        assert eta_bep > eta_low, "BEP efficiency should exceed low flow"
        assert eta_bep > eta_high, "BEP efficiency should exceed high flow"
        assert abs(eta_bep - 0.70) < 0.01, "BEP efficiency should match spec"

    def test_power_consumption_realistic(self):
        """Power consumption should be realistic"""
        pump = PumpWithCurve(
            "Test",
            rated_flow=0.05,  # 50 g/s
            rated_head=5.0,   # 5 meters
            efficiency_bep=0.70
        )

        result = pump.update(60.0, {'speed': 1.0})

        # Theoretical power: P = ρ * g * Q * H / η
        # ρ = 1000 kg/m³, g = 9.81 m/s², Q = 0.05/1000 = 0.00005 m³/s, H = 5m, η = 0.7
        # P ≈ 1000 * 9.81 * 0.00005 * 5 / 0.7 ≈ 3.5 W
        expected_power = 1000 * 9.81 * (0.05/1000) * 5.0 / 0.70

        assert abs(result['power'] - expected_power) < 1.0, \
            f"Power {result['power']:.1f}W should be near {expected_power:.1f}W"

    def test_zero_power_at_zero_speed(self):
        """No power consumption when pump is off"""
        pump = PumpWithCurve("Test")

        result = pump.update(60.0, {'speed': 0.0})

        assert result['power'] == 0.0, "Power should be zero when pump off"
        assert result['flow_rate'] == 0.0, "Flow should be zero when pump off"

    def test_affinity_laws_speed_scaling(self):
        """Flow should scale linearly with speed (affinity laws)"""
        pump = PumpWithCurve("Test", rated_flow=0.05)

        result_half = pump.update(60.0, {'speed': 0.5})
        result_full = pump.update(60.0, {'speed': 1.0})

        # Flow should scale linearly with speed
        flow_ratio = result_full['flow_rate'] / result_half['flow_rate']

        assert abs(flow_ratio - 2.0) < 0.1, \
            f"Flow ratio {flow_ratio} should be ~2.0 for 2x speed increase"


class TestPumpIntegration:
    """Test pump in integrated scenarios"""

    def test_pump_efficiency_under_varying_load(self):
        """Efficiency should vary realistically under different loads"""
        pump = PumpWithCurve(
            "Test",
            rated_flow=0.05,
            rated_head=5.0,
            efficiency_bep=0.70
        )

        speeds = [0.2, 0.5, 0.8, 1.0]
        efficiencies = []

        for speed in speeds:
            if speed > 0:
                result = pump.update(60.0, {'speed': speed})
                efficiencies.append(result['efficiency'])

        # All efficiencies should be positive and less than 1
        assert all(0 < eta < 1.0 for eta in efficiencies), \
            "All efficiencies should be between 0 and 1"

        # Efficiency should generally increase towards rated speed
        assert efficiencies[-1] > efficiencies[0], \
            "Efficiency at rated speed should exceed low speed"

    def test_energy_consumption_accumulation(self):
        """Test cumulative energy consumption over time"""
        pump = PumpWithCurve("Test", rated_flow=0.05, rated_head=5.0)

        dt = 60.0  # seconds
        total_energy = 0.0

        # Run pump for 1 hour
        for _ in range(60):
            result = pump.update(dt, {'speed': 1.0})
            total_energy += result['power'] * dt  # Joules

        # Convert to kWh
        total_kwh = total_energy / 3.6e6

        # Should be small but measurable (few Wh for small pump)
        assert 0 < total_kwh < 0.1, \
            f"Energy consumption {total_kwh:.4f} kWh should be realistic for 1 hour"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
