"""
Example: Upgrading Components Independently

This demonstrates how to improve physics models or control logic
without modifying the system integration code.
"""

import numpy as np
from components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    ThermalLoad, ThermalLoadParams,
    FluidComponent
)
from controller import Controller
from modular_simulation import ModularThermalSystem
from typing import Dict, Any


# ============================================================================
# UPGRADE EXAMPLE 1: Improved Controller with Safety Features
# ============================================================================

class ImprovedController(Controller):
    """
    Enhanced controller with:
    - Better over-temperature protection
    - Adaptive flow control
    - Load prioritization
    """

    def __init__(
        self,
        name: str = "ImprovedController",
        solar_dT_on: float = 7.0,
        solar_dT_off: float = 3.0,
        tank_max_temp: float = 75.0,
        tank_critical_temp: float = 70.0,
        min_useful_temp: float = 40.0
    ):
        super().__init__(name)
        self.solar_dT_on = solar_dT_on
        self.solar_dT_off = solar_dT_off
        self.tank_max_temp = tank_max_temp
        self.tank_critical_temp = tank_critical_temp
        self.min_useful_temp = min_useful_temp

        self.solar_pump_on = False
        self.load_pump_on = True

    def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced control logic with safety features"""
        T_collector = system_state.get('T_collector', 20.0)
        T_tank = system_state.get('T_tank', 20.0)
        Q_demand = system_state.get('Q_demand', 0.0)
        irradiance = system_state.get('irradiance', 0.0)

        dT = T_collector - T_tank

        # === SOLAR LOOP CONTROL ===
        # Safety: Never run solar pump if tank is too hot
        if T_tank >= self.tank_max_temp:
            self.solar_pump_on = False
            solar_pump_speed = 0.0

        # Normal operation with hysteresis
        elif not self.solar_pump_on:
            if dT > self.solar_dT_on and irradiance > 150:
                self.solar_pump_on = True
        else:
            if dT < self.solar_dT_off or irradiance < 50:
                self.solar_pump_on = False

        if self.solar_pump_on:
            # Adaptive speed: higher when tank is cooler (more useful)
            # Lower speed when approaching max temp
            if T_tank < self.tank_critical_temp:
                # Normal operation: speed based on dT
                solar_pump_speed = np.clip(0.4 + (dT - self.solar_dT_on) / 15.0, 0.4, 1.0)
            else:
                # Approaching limit: reduce speed
                temp_factor = 1.0 - (T_tank - self.tank_critical_temp) / (self.tank_max_temp - self.tank_critical_temp)
                solar_pump_speed = np.clip(temp_factor, 0.2, 0.5)
        else:
            solar_pump_speed = 0.0

        # === LOAD LOOP CONTROL ===
        # Only run load pump if tank is hot enough to be useful
        if Q_demand > 0 and T_tank > self.min_useful_temp:
            # Calculate required flow rate
            fluid_cp = 4186.0
            dT_load = max(10.0, T_tank - 35.0)  # Adaptive based on tank temp
            required_flow = Q_demand / (fluid_cp * dT_load)

            # Speed based on demand
            load_pump_speed = np.clip(required_flow / 0.05, 0.15, 1.0)
            self.load_pump_on = True
        else:
            # Minimal circulation to prevent stratification issues
            load_pump_speed = 0.05
            self.load_pump_on = False

        return {
            'solar_pump_speed': solar_pump_speed,
            'load_pump_speed': load_pump_speed,
            'solar_valve_position': 1.0 if self.solar_pump_on else 0.0,
            'load_valve_position': 1.0 if self.load_pump_on else 0.0,
            'control_state': {
                'solar_pump_on': self.solar_pump_on,
                'load_pump_on': self.load_pump_on,
                'dT_solar': dT,
                'safety_limit_active': T_tank >= self.tank_max_temp
            }
        }


# ============================================================================
# UPGRADE EXAMPLE 2: Enhanced Solar Collector with IAM
# ============================================================================

class EnhancedSolarCollector(SolarCollector):
    """
    Enhanced collector with:
    - Incidence Angle Modifier (IAM)
    - Temperature-dependent efficiency
    - More accurate heat loss model
    """

    def __init__(self, name: str, params: SolarCollectorParams):
        super().__init__(name, params)
        # Additional parameters for enhanced model
        self.K_theta_1 = 0.15  # IAM coefficient 1
        self.K_theta_2 = 0.05  # IAM coefficient 2

    def calculate_IAM(self, time_hours: float) -> float:
        """
        Calculate Incidence Angle Modifier based on time of day.
        Simplified model - real version would use sun position calculations.
        """
        # Approximate incidence angle based on time from solar noon
        solar_noon = 12.0
        hour_angle = abs(time_hours - solar_noon)

        # Simplified IAM (real: K_theta(θ) = 1 - K1*(1/cos(θ) - 1) - K2*(1/cos(θ) - 1)²)
        if hour_angle < 6:
            theta_deg = hour_angle * 15  # Rough approximation
            theta_rad = np.deg2rad(theta_deg)
            cos_theta = np.cos(theta_rad)
            IAM = 1.0 - self.K_theta_1 * (1/cos_theta - 1)
            return max(IAM, 0.5)  # Minimum 0.5
        return 0.5

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced update with IAM and temperature-dependent efficiency"""
        T_inlet = inputs.get('T_inlet', 20.0)
        flow_rate = inputs.get('flow_rate', 0.0)
        irradiance = inputs.get('irradiance', 0.0)
        T_ambient = inputs.get('T_ambient', 20.0)
        fluid_cp = inputs.get('fluid_cp', 3500.0)
        time_hours = inputs.get('time_hours', 12.0)

        # Calculate IAM
        IAM = self.calculate_IAM(time_hours % 24)

        # Temperature-dependent efficiency
        T_avg = (self.T_collector + T_inlet) / 2
        dT_op = T_avg - T_ambient
        efficiency = self.params.efficiency - (self.params.heat_loss_coef * dT_op / irradiance if irradiance > 0 else 0)
        efficiency = max(efficiency, 0.1)  # Minimum efficiency

        # Heat absorbed with IAM
        Q_absorbed = (irradiance * self.params.area *
                     self.params.absorptance * efficiency * IAM)

        # Enhanced heat loss (radiation + convection)
        Q_loss_conv = self.params.heat_loss_coef * self.params.area * dT_op
        Q_loss_rad = 5.67e-8 * self.params.area * self.params.absorptance * (
            (self.T_collector + 273.15)**4 - (T_ambient + 273.15)**4
        )
        Q_loss = Q_loss_conv + Q_loss_rad

        Q_net = Q_absorbed - Q_loss

        # Update collector temperature
        dT_collector = (Q_net * dt) / (self.params.thermal_mass * self.params.specific_heat)
        self.T_collector += dT_collector

        # Calculate outlet temperature
        T_outlet = self.calculate_outlet_temp(T_inlet, flow_rate, dt)
        Q_to_fluid = flow_rate * fluid_cp * (T_outlet - T_inlet) if flow_rate > 0 else 0

        return {
            'T_outlet': T_outlet,
            'Q_collected': Q_net,
            'Q_to_fluid': Q_to_fluid,
            'T_collector': self.T_collector,
            'IAM': IAM,
            'efficiency': efficiency
        }


# ============================================================================
# DEMONSTRATION: Run with improved components
# ============================================================================

def main():
    """Demonstrate component upgrades"""

    print("="*70)
    print("COMPONENT UPGRADE DEMONSTRATION")
    print("="*70)
    print("\nRunning simulation with IMPROVED components...")
    print("  • Enhanced Controller (better safety, adaptive control)")
    print("  • Standard components for tank and load")
    print()

    # Use IMPROVED collector (optional - can use either)
    collector = EnhancedSolarCollector(
        "EnhancedSolarCollector",
        SolarCollectorParams(area=2.5, efficiency=0.75)
    )

    # Standard tank
    tank = StorageTank(
        "StorageTank",
        StorageTankParams(volume=0.2, mass=200.0),
        initial_temp=30.0
    )

    # Standard load with variable demand
    load = ThermalLoad(
        "BuildingLoad",
        ThermalLoadParams(
            base_load=300.0,
            peak_load=1500.0,
            load_profile='variable'
        )
    )

    # IMPROVED controller
    controller = ImprovedController(
        solar_dT_on=7.0,
        solar_dT_off=3.0,
        tank_max_temp=75.0,
        tank_critical_temp=68.0
    )

    # Create system (SAME integration code!)
    system = ModularThermalSystem(
        collector=collector,
        tank=tank,
        load=load,
        controller=controller,
        ambient_temp=20.0
    )

    # Run simulation
    system.run_simulation(duration_hours=48.0, dt=60.0)

    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    print("\nComparison to basic controller:")
    print("  - Improved safety features (temperature limiting)")
    print("  - Better load matching through adaptive flow control")
    print("  - More efficient solar collection with IAM model")
    print("  - Demonstrates modular upgrade capability")
    print("\nNo changes to system integration code required!")
    print("="*70)

    # Plot results
    system.plot_results()


if __name__ == "__main__":
    main()
