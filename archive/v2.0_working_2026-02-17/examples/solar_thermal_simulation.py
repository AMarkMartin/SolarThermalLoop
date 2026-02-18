"""
Solar Thermal System Simulation
Simulates heat transfer from a solar panel to a storage tank via a pumped fluid loop.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class FluidProperties:
    """Properties of the heat transfer fluid (water/glycol mixture)"""
    density: float = 1040.0  # kg/m³ (50% glycol)
    specific_heat: float = 3500.0  # J/(kg·K)
    viscosity: float = 0.004  # Pa·s


@dataclass
class SolarPanelParams:
    """Solar panel parameters"""
    area: float = 2.0  # m²
    efficiency: float = 0.75  # Collector efficiency (0-1)
    absorptance: float = 0.95  # Solar absorptance
    heat_loss_coef: float = 5.0  # W/(m²·K) - combined convection/radiation loss


@dataclass
class StorageTankParams:
    """Storage tank parameters"""
    volume: float = 0.15  # m³ (150 liters)
    mass: float = 150.0  # kg (assuming water)
    specific_heat: float = 4186.0  # J/(kg·K)
    surface_area: float = 1.5  # m² (for heat loss calculation)
    heat_loss_coef: float = 1.5  # W/(m²·K) - insulated tank


class SolarThermalSystem:
    def __init__(
        self,
        fluid: FluidProperties,
        panel: SolarPanelParams,
        tank: StorageTankParams,
        flow_rate: float = 0.02,  # kg/s
        ambient_temp: float = 20.0,  # °C
    ):
        self.fluid = fluid
        self.panel = panel
        self.tank = tank
        self.flow_rate = flow_rate  # Mass flow rate (kg/s)
        self.T_ambient = ambient_temp  # Ambient temperature (°C)

        # Initial conditions
        self.T_tank = ambient_temp  # Tank temperature (°C)
        self.T_panel_out = ambient_temp  # Panel outlet temperature (°C)

        # History for plotting
        self.time_history = []
        self.T_tank_history = []
        self.T_panel_out_history = []
        self.Q_solar_history = []
        self.Q_tank_history = []

    def solar_irradiance(self, time_hours: float) -> float:
        """
        Calculate solar irradiance based on time of day (W/m²)
        Simplified sinusoidal model for daytime hours
        """
        if 6 <= time_hours <= 18:
            # Peak at solar noon (12:00)
            angle = np.pi * (time_hours - 6) / 12
            return 1000 * np.sin(angle)  # Peak 1000 W/m²
        return 0.0

    def calculate_panel_heat_gain(self, irradiance: float, T_panel_in: float) -> tuple:
        """
        Calculate heat gained by the solar panel and outlet temperature.

        Returns:
            Q_solar: Heat power absorbed (W)
            T_panel_out: Outlet fluid temperature (°C)
        """
        # Heat absorbed from solar radiation
        Q_absorbed = irradiance * self.panel.area * self.panel.absorptance * self.panel.efficiency

        # Heat loss from panel to ambient (based on average panel temperature)
        T_panel_avg = (T_panel_in + self.T_panel_out) / 2
        Q_loss = self.panel.heat_loss_coef * self.panel.area * (T_panel_avg - self.T_ambient)

        # Net heat gain
        Q_solar = Q_absorbed - Q_loss

        # Calculate outlet temperature using energy balance
        # Q = m_dot * cp * (T_out - T_in)
        if self.flow_rate > 0:
            delta_T = Q_solar / (self.flow_rate * self.fluid.specific_heat)
            T_panel_out = T_panel_in + delta_T
        else:
            T_panel_out = T_panel_in

        return Q_solar, T_panel_out

    def calculate_tank_heat_transfer(self, T_fluid_in: float, dt: float) -> float:
        """
        Calculate heat transfer to the storage tank and update tank temperature.

        Args:
            T_fluid_in: Inlet fluid temperature from panel (°C)
            dt: Time step (seconds)

        Returns:
            Q_to_tank: Heat transferred to tank (W)
        """
        # Heat delivered by the fluid to the tank
        # Assuming perfect heat exchange (fluid exits at tank temperature)
        T_fluid_out = self.T_tank
        Q_delivered = self.flow_rate * self.fluid.specific_heat * (T_fluid_in - T_fluid_out)

        # Heat loss from tank to ambient
        Q_tank_loss = self.tank.heat_loss_coef * self.tank.surface_area * (self.T_tank - self.T_ambient)

        # Net heat to tank
        Q_to_tank = Q_delivered - Q_tank_loss

        # Update tank temperature using energy balance
        # Q * dt = m * cp * dT
        dT_tank = (Q_to_tank * dt) / (self.tank.mass * self.tank.specific_heat)
        self.T_tank += dT_tank

        return Q_to_tank

    def simulate_timestep(self, time_hours: float, dt: float):
        """
        Simulate one time step of the system.

        Args:
            time_hours: Current simulation time (hours)
            dt: Time step (seconds)
        """
        # Get solar irradiance
        irradiance = self.solar_irradiance(time_hours)

        # Fluid enters panel at tank temperature (return loop)
        T_panel_in = self.T_tank

        # Calculate panel heat gain and outlet temperature
        Q_solar, self.T_panel_out = self.calculate_panel_heat_gain(irradiance, T_panel_in)

        # Calculate heat transfer to tank
        Q_to_tank = self.calculate_tank_heat_transfer(self.T_panel_out, dt)

        # Store history
        self.time_history.append(time_hours)
        self.T_tank_history.append(self.T_tank)
        self.T_panel_out_history.append(self.T_panel_out)
        self.Q_solar_history.append(Q_solar)
        self.Q_tank_history.append(Q_to_tank)

    def run_simulation(self, duration_hours: float = 24.0, dt: float = 60.0):
        """
        Run the full simulation.

        Args:
            duration_hours: Total simulation time (hours)
            dt: Time step (seconds)
        """
        num_steps = int(duration_hours * 3600 / dt)

        print(f"Running simulation for {duration_hours} hours...")
        print(f"Initial tank temperature: {self.T_tank:.1f}°C")
        print(f"Ambient temperature: {self.T_ambient:.1f}°C")
        print(f"Flow rate: {self.flow_rate:.3f} kg/s")
        print(f"Panel area: {self.panel.area} m²")
        print(f"Tank volume: {self.tank.volume * 1000:.0f} liters\n")

        for step in range(num_steps):
            time_hours = step * dt / 3600
            self.simulate_timestep(time_hours, dt)

        print(f"Simulation complete!")
        print(f"Final tank temperature: {self.T_tank:.1f}°C")
        print(f"Temperature rise: {self.T_tank - self.T_tank_history[0]:.1f}°C")

        # Calculate total energy collected
        total_energy = np.trapezoid(self.Q_solar_history,
                                    np.array(self.time_history) * 3600) / 1e6  # MJ
        print(f"Total solar energy collected: {total_energy:.1f} MJ")

    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Solar Thermal System Simulation Results', fontsize=14, fontweight='bold')

        # Temperature plot
        ax1 = axes[0, 0]
        ax1.plot(self.time_history, self.T_tank_history, 'r-', linewidth=2, label='Tank Temperature')
        ax1.plot(self.time_history, self.T_panel_out_history, 'b-', linewidth=2, label='Panel Outlet Temperature')
        ax1.axhline(y=self.T_ambient, color='gray', linestyle='--', label='Ambient Temperature')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('System Temperatures')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Solar irradiance and heat collection
        ax2 = axes[0, 1]
        irradiance_history = [self.solar_irradiance(t) for t in self.time_history]
        ax2_twin = ax2.twinx()

        ax2.plot(self.time_history, irradiance_history, 'orange', linewidth=2, label='Solar Irradiance')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Solar Irradiance (W/m²)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax2_twin.plot(self.time_history, self.Q_solar_history, 'g-', linewidth=2, label='Heat Collected')
        ax2_twin.set_ylabel('Heat Power (W)', color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.set_title('Solar Input and Heat Collection')
        ax2.grid(True, alpha=0.3)

        # Heat transfer to tank
        ax3 = axes[1, 0]
        ax3.plot(self.time_history, self.Q_tank_history, 'm-', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Heat Transfer to Tank (W)')
        ax3.set_title('Net Heat Transfer to Storage Tank')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Energy balance
        ax4 = axes[1, 1]
        cumulative_solar = np.cumsum(self.Q_solar_history) * (self.time_history[1] - self.time_history[0]) * 3600 / 1e6
        cumulative_tank = np.cumsum(self.Q_tank_history) * (self.time_history[1] - self.time_history[0]) * 3600 / 1e6

        ax4.plot(self.time_history, cumulative_solar, 'g-', linewidth=2, label='Cumulative Solar Collection')
        ax4.plot(self.time_history, cumulative_tank, 'm-', linewidth=2, label='Cumulative Tank Heat Gain')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Energy (MJ)')
        ax4.set_title('Cumulative Energy Balance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('solar_thermal_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'solar_thermal_results.png'")
        plt.show()


def main():
    """Run the solar thermal system simulation"""

    # Initialize system components
    fluid = FluidProperties()
    panel = SolarPanelParams(area=2.0)
    tank = StorageTankParams(volume=0.15)

    # Create system with specified flow rate
    system = SolarThermalSystem(
        fluid=fluid,
        panel=panel,
        tank=tank,
        flow_rate=0.02,  # 20 g/s = 72 kg/hr
        ambient_temp=20.0
    )

    # Run simulation for 24 hours with 1-minute time steps
    system.run_simulation(duration_hours=24.0, dt=60.0)

    # Plot results
    system.plot_results()

    # Print thermodynamic efficiency analysis
    print("\n" + "="*60)
    print("THERMODYNAMIC ANALYSIS")
    print("="*60)

    max_tank_temp = max(system.T_tank_history)
    avg_solar_power = np.mean([q for q in system.Q_solar_history if q > 0])

    print(f"Maximum tank temperature reached: {max_tank_temp:.1f}°C")
    print(f"Average solar collection rate: {avg_solar_power:.1f} W")

    # Calculate system efficiency
    total_incident = np.trapezoid(
        [system.solar_irradiance(t) * panel.area for t in system.time_history],
        np.array(system.time_history) * 3600
    ) / 1e6
    total_collected = np.trapezoid(system.Q_solar_history, np.array(system.time_history) * 3600) / 1e6

    if total_incident > 0:
        efficiency = (total_collected / total_incident) * 100
        print(f"Overall system efficiency: {efficiency:.1f}%")

    print("="*60)


if __name__ == "__main__":
    main()
