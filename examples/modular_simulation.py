"""
Modular Solar Thermal System Simulation
Demonstrates component-based architecture with control system and thermal load.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    ThermalLoad, ThermalLoadParams,
    Valve, Pump, Component
)
from src.control import BasicController


class ModularThermalSystem:
    """
    Integrated solar thermal system using modular components.
    Components can be upgraded independently while maintaining system integration.
    """

    def __init__(
        self,
        collector: SolarCollector,
        tank: StorageTank,
        load: ThermalLoad,
        controller: BasicController,
        ambient_temp: float = 20.0
    ):
        # Components
        self.collector = collector
        self.tank = tank
        self.load = load
        self.controller = controller

        # Pumps and valves
        self.solar_pump = Pump("SolarPump", max_flow=0.05)  # 50 g/s = 180 kg/hr
        self.load_pump = Pump("LoadPump", max_flow=0.05)
        self.solar_valve = Valve("SolarValve", max_flow=0.05)
        self.load_valve = Valve("LoadValve", max_flow=0.05)

        # Environment
        self.T_ambient = ambient_temp

        # Fluid properties
        self.fluid_density = 1040.0  # kg/m³
        self.fluid_cp_solar = 3500.0  # J/(kg·K) - glycol mixture
        self.fluid_cp_load = 4186.0  # J/(kg·K) - water

        # History for logging
        self.history: Dict[str, List] = {
            'time': [],
            'T_tank': [],
            'T_collector': [],
            'T_load_supply': [],
            'T_load_return': [],
            'Q_solar': [],
            'Q_load_demand': [],
            'Q_load_actual': [],
            'solar_pump_speed': [],
            'load_pump_speed': [],
            'irradiance': [],
            'energy_stored': []
        }

    def solar_irradiance(self, time_hours: float) -> float:
        """Calculate solar irradiance based on time of day"""
        if 6 <= time_hours <= 18:
            angle = np.pi * (time_hours - 6) / 12
            return 1000 * np.sin(angle)  # Peak 1000 W/m²
        return 0.0

    def simulate_timestep(self, time_hours: float, dt: float):
        """
        Simulate one timestep with modular component updates.

        System flow:
        1. Get current system state
        2. Controller computes control actions
        3. Update pumps/valves based on control
        4. Update each component with new inputs
        5. Log results
        """
        # Current solar irradiance
        irradiance = self.solar_irradiance(time_hours)

        # Get current system state for controller
        system_state = {
            'T_collector': self.collector.T_collector,
            'T_tank': self.tank.T_tank,
            'Q_demand': self.load.Q_demand,
            'irradiance': irradiance,
            'time_hours': time_hours
        }

        # Compute control actions
        control = self.controller.compute_control(system_state)

        # Update pumps based on control
        solar_pump_out = self.solar_pump.update(dt, {'speed': control['solar_pump_speed']})
        load_pump_out = self.load_pump.update(dt, {'speed': control['load_pump_speed']})

        # Get flow rates
        flow_rate_solar = solar_pump_out['flow_rate']
        flow_rate_load = load_pump_out['flow_rate']

        # Update valves (currently just mirror pump state)
        self.solar_valve.update(dt, {'position': control['solar_valve_position']})
        self.load_valve.update(dt, {'position': control['load_valve_position']})

        # === Solar Loop ===
        # Collector inlet comes from tank
        collector_inputs = {
            'T_inlet': self.tank.T_tank,
            'flow_rate': flow_rate_solar,
            'irradiance': irradiance,
            'T_ambient': self.T_ambient,
            'fluid_cp': self.fluid_cp_solar
        }
        collector_out = self.collector.update(dt, collector_inputs)

        # === Load Loop ===
        # Load inlet comes from tank
        load_inputs = {
            'T_inlet': self.tank.T_tank,
            'flow_rate': flow_rate_load,
            'time_hours': time_hours,
            'fluid_cp': self.fluid_cp_load
        }
        load_out = self.load.update(dt, load_inputs)

        # === Storage Tank ===
        # Tank receives hot fluid from collector and cold fluid from load
        tank_inputs = {
            'T_inlet_solar': collector_out['T_outlet'],
            'flow_rate_solar': flow_rate_solar,
            'T_inlet_load': load_out['T_outlet'],
            'flow_rate_load': -flow_rate_load,  # Negative because it's extracting heat
            'T_ambient': self.T_ambient,
            'fluid_cp': self.fluid_cp_solar
        }
        tank_out = self.tank.update(dt, tank_inputs)

        # Log data
        self.history['time'].append(time_hours)
        self.history['T_tank'].append(tank_out['T_tank'])
        self.history['T_collector'].append(collector_out['T_collector'])
        self.history['T_load_supply'].append(self.tank.T_tank)
        self.history['T_load_return'].append(load_out['T_outlet'])
        self.history['Q_solar'].append(collector_out['Q_to_fluid'])
        self.history['Q_load_demand'].append(load_out['Q_demand'])
        self.history['Q_load_actual'].append(load_out['Q_actual'])
        self.history['solar_pump_speed'].append(solar_pump_out['speed'])
        self.history['load_pump_speed'].append(load_pump_out['speed'])
        self.history['irradiance'].append(irradiance)
        self.history['energy_stored'].append(tank_out['energy_stored'])

    def run_simulation(self, duration_hours: float = 48.0, dt: float = 60.0):
        """Run the simulation"""
        num_steps = int(duration_hours * 3600 / dt)

        print(f"Running modular simulation for {duration_hours} hours...")
        print(f"Initial tank temperature: {self.tank.T_tank:.1f}°C")
        print(f"Ambient temperature: {self.T_ambient:.1f}°C")
        print(f"Thermal load: {self.load.params.load_profile}")
        print(f"Controller: {self.controller.name}\n")

        for step in range(num_steps):
            time_hours = step * dt / 3600
            self.simulate_timestep(time_hours, dt)

            # Progress indicator
            if step % (num_steps // 10) == 0:
                progress = 100 * step / num_steps
                print(f"Progress: {progress:.0f}% - Tank: {self.tank.T_tank:.1f}°C")

        print(f"\nSimulation complete!")
        print(f"Final tank temperature: {self.tank.T_tank:.1f}°C")
        print(f"Max tank temperature: {max(self.history['T_tank']):.1f}°C")

        # Energy analysis
        total_solar_energy = np.trapezoid(self.history['Q_solar'],
                                          np.array(self.history['time']) * 3600) / 1e6
        total_load_demand = np.trapezoid(self.history['Q_load_demand'],
                                         np.array(self.history['time']) * 3600) / 1e6
        total_load_served = np.trapezoid(self.history['Q_load_actual'],
                                         np.array(self.history['time']) * 3600) / 1e6

        print(f"\nEnergy Summary:")
        print(f"  Solar energy collected: {total_solar_energy:.1f} MJ")
        print(f"  Load demand: {total_load_demand:.1f} MJ")
        print(f"  Load served: {total_load_served:.1f} MJ")
        if total_load_demand > 0:
            print(f"  Load fraction met: {100 * total_load_served / total_load_demand:.1f}%")

    def plot_results(self):
        """Plot comprehensive system results"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Modular Solar Thermal System - Results', fontsize=14, fontweight='bold')

        time = self.history['time']

        # 1. Temperatures
        ax1 = axes[0, 0]
        ax1.plot(time, self.history['T_tank'], 'r-', linewidth=2, label='Tank')
        ax1.plot(time, self.history['T_collector'], 'orange', linewidth=2, label='Collector')
        ax1.plot(time, self.history['T_load_supply'], 'b-', linewidth=1.5, label='Load Supply')
        ax1.plot(time, self.history['T_load_return'], 'c-', linewidth=1.5, label='Load Return')
        ax1.axhline(y=self.T_ambient, color='gray', linestyle='--', label='Ambient')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('System Temperatures')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Solar collection
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        ax2.fill_between(time, 0, self.history['irradiance'],
                         color='yellow', alpha=0.3, label='Irradiance')
        ax2.set_ylabel('Irradiance (W/m²)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2_twin.plot(time, np.array(self.history['Q_solar']) / 1000,
                     'g-', linewidth=2, label='Heat Collected')
        ax2_twin.set_ylabel('Heat Power (kW)', color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.set_xlabel('Time (hours)')
        ax2.set_title('Solar Collection')
        ax2.grid(True, alpha=0.3)

        # 3. Load demand vs actual
        ax3 = axes[1, 0]
        ax3.plot(time, np.array(self.history['Q_load_demand']) / 1000,
                'r--', linewidth=2, label='Demand')
        ax3.plot(time, np.array(self.history['Q_load_actual']) / 1000,
                'b-', linewidth=2, label='Actual')
        ax3.fill_between(time,
                        np.array(self.history['Q_load_actual']) / 1000,
                        np.array(self.history['Q_load_demand']) / 1000,
                        where=np.array(self.history['Q_load_actual']) < np.array(self.history['Q_load_demand']),
                        color='red', alpha=0.2, label='Unmet')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Thermal Power (kW)')
        ax3.set_title('Load: Demand vs Served')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Pump speeds
        ax4 = axes[1, 1]
        ax4.plot(time, np.array(self.history['solar_pump_speed']) * 100,
                'g-', linewidth=2, label='Solar Pump')
        ax4.plot(time, np.array(self.history['load_pump_speed']) * 100,
                'b-', linewidth=2, label='Load Pump')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Pump Speed (%)')
        ax4.set_title('Pump Control Signals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([-5, 105])

        # 5. Energy stored
        ax5 = axes[2, 0]
        ax5.plot(time, self.history['energy_stored'], 'm-', linewidth=2)
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Energy Stored (MJ)')
        ax5.set_title('Tank Energy Storage')
        ax5.grid(True, alpha=0.3)

        # 6. Cumulative energy
        ax6 = axes[2, 1]
        cum_solar = np.cumsum(self.history['Q_solar']) * (time[1] - time[0]) * 3600 / 1e6
        cum_load_demand = np.cumsum(self.history['Q_load_demand']) * (time[1] - time[0]) * 3600 / 1e6
        cum_load_actual = np.cumsum(self.history['Q_load_actual']) * (time[1] - time[0]) * 3600 / 1e6

        ax6.plot(time, cum_solar, 'g-', linewidth=2, label='Solar Collected')
        ax6.plot(time, cum_load_demand, 'r--', linewidth=2, label='Load Demand')
        ax6.plot(time, cum_load_actual, 'b-', linewidth=2, label='Load Served')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('Cumulative Energy (MJ)')
        ax6.set_title('Cumulative Energy Balance')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('modular_system_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'modular_system_results.png'")
        plt.show()


def main():
    """Run the modular simulation with thermal load and control"""

    # Create components with parameters
    collector = SolarCollector(
        "SolarCollector",
        SolarCollectorParams(area=2.5, efficiency=0.75)
    )

    tank = StorageTank(
        "StorageTank",
        StorageTankParams(volume=0.2, mass=200.0),
        initial_temp=30.0  # Start with some stored heat
    )

    # Create thermal load with variable demand profile
    load = ThermalLoad(
        "BuildingLoad",
        ThermalLoadParams(
            base_load=300.0,      # 300 W base
            peak_load=1500.0,     # 1.5 kW peak
            load_profile='variable'  # Variable demand
        )
    )

    # Create controller
    controller = BasicController(
        solar_dT_threshold=5.0,
        tank_max_temp=80.0
    )

    # Create system
    system = ModularThermalSystem(
        collector=collector,
        tank=tank,
        load=load,
        controller=controller,
        ambient_temp=20.0
    )

    # Run simulation for 48 hours to see full charging/discharging cycles
    system.run_simulation(duration_hours=48.0, dt=60.0)

    # Plot results
    system.plot_results()

    # Demonstrate component upgradability
    print("\n" + "="*60)
    print("COMPONENT MODULARITY DEMONSTRATION")
    print("="*60)
    print("\nEach component can be independently upgraded:")
    print("  • Solar Collector: Add angle-dependent efficiency, IAM model")
    print("  • Storage Tank: Upgrade to stratified (multi-node) model")
    print("  • Thermal Load: Add realistic building thermal model")
    print("  • Controller: Upgrade to PID, MPC, or ML-based control")
    print("  • Valves/Pumps: Add detailed hydraulics, efficiency curves")
    print("\nAll upgrades maintain the same interface and integration!")
    print("="*60)


if __name__ == "__main__":
    main()
