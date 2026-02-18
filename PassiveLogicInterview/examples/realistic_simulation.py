"""
Realistic Solar Thermal System Simulation
Featuring:
- Realistic solar radiation with sun position
- Weather forecasting
- Pump performance curves
- Building thermal mass model
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from datetime import datetime

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    Valve
)
from src.models import (
    SolarRadiationModel, LocationParams, WeatherForecast,
    PumpWithCurve,
    BuildingThermalMass, BuildingParams
)
from src.control import BasicController


class RealisticThermalSystem:
    """
    Complete solar thermal system with realistic physics:
    - Geographic-aware solar radiation
    - Weather effects
    - Pump curves with efficiency
    - Building thermal mass
    """

    def __init__(
        self,
        collector: SolarCollector,
        tank: StorageTank,
        building: BuildingThermalMass,
        controller: BasicController,
        location: LocationParams,
        start_day: int = 180,  # Day of year (180 = ~June 29)
        ambient_temp: float = 20.0
    ):
        # Components
        self.collector = collector
        self.tank = tank
        self.building = building
        self.controller = controller

        # Pumps with realistic curves
        self.solar_pump = PumpWithCurve(
            "SolarPump",
            rated_flow=0.04,  # 40 g/s at BEP
            rated_head=4.0,
            max_head=6.0,
            efficiency_bep=0.68
        )
        self.load_pump = PumpWithCurve(
            "LoadPump",
            rated_flow=0.05,
            rated_head=5.0,
            max_head=7.5,
            efficiency_bep=0.70
        )

        # Valves
        self.solar_valve = Valve("SolarValve", max_flow=0.05)
        self.load_valve = Valve("LoadValve", max_flow=0.05)

        # Environment and solar model
        self.location = location
        self.solar_model = SolarRadiationModel(location)
        self.weather = WeatherForecast(days=3)
        self.start_day = start_day
        self.T_ambient_base = ambient_temp

        # Fluid properties
        self.fluid_cp_solar = 3500.0  # J/(kg·K) - glycol
        self.fluid_cp_load = 4186.0  # J/(kg·K) - water

        # History
        self.history: Dict[str, List] = {
            'time': [],
            'day_of_year': [],
            'T_tank': [],
            'T_collector': [],
            'T_building': [],
            'T_ambient': [],
            'irradiance_total': [],
            'irradiance_direct': [],
            'irradiance_diffuse': [],
            'solar_altitude': [],
            'cloud_cover': [],
            'Q_solar_collected': [],
            'Q_building_demand': [],
            'Q_building_delivered': [],
            'Q_building_loss': [],
            'Q_building_solar_gain': [],
            'solar_pump_speed': [],
            'solar_pump_power': [],
            'solar_pump_efficiency': [],
            'load_pump_speed': [],
            'load_pump_power': [],
            'load_pump_efficiency': [],
            'heating_active': [],
        }

    def get_ambient_temperature(self, time_hours: float) -> float:
        """Get ambient temperature with diurnal variation and forecast"""
        hour = int(time_hours)
        T_forecast = self.weather.get_temperature(hour)

        # Diurnal variation
        T_diurnal = self.T_ambient_base - 5 * np.cos(2 * np.pi * (time_hours - 15) / 24)

        # Blend forecast with pattern
        return 0.7 * T_forecast + 0.3 * T_diurnal

    def simulate_timestep(self, time_hours: float, dt: float):
        """Simulate one timestep with all enhanced models"""

        # Current day of year
        day_of_year = self.start_day + int(time_hours / 24)
        hour_of_day = time_hours % 24

        # Get ambient temperature
        T_ambient = self.get_ambient_temperature(time_hours)

        # Get cloud cover from forecast
        cloud_cover = self.weather.get_cloud_cover(int(time_hours))

        # Calculate realistic solar irradiance
        solar_data = self.solar_model.calculate_irradiance(
            time_hours=hour_of_day,
            day_of_year=day_of_year,
            cloud_cover=cloud_cover,
            panel_tilt=40.0,  # 40° tilt
            panel_azimuth=180.0  # Facing south
        )

        irradiance = solar_data['total']
        solar_altitude = solar_data['altitude']

        # === CONTROL SYSTEM ===
        system_state = {
            'T_collector': self.collector.T_collector,
            'T_tank': self.tank.T_tank,
            'Q_demand': self.building.Q_heating_required,
            'irradiance': irradiance,
            'time_hours': time_hours
        }

        control = self.controller.compute_control(system_state)

        # === UPDATE PUMPS ===
        solar_pump_out = self.solar_pump.update(dt, {'speed': control['solar_pump_speed']})
        load_pump_out = self.load_pump.update(dt, {'speed': control['load_pump_speed']})

        flow_solar = solar_pump_out['flow_rate']
        flow_load = load_pump_out['flow_rate']

        # === UPDATE VALVES ===
        self.solar_valve.update(dt, {'position': control['solar_valve_position']})
        self.load_valve.update(dt, {'position': control['load_valve_position']})

        # === SOLAR COLLECTOR ===
        collector_inputs = {
            'T_inlet': self.tank.T_tank,
            'flow_rate': flow_solar,
            'irradiance': irradiance,
            'T_ambient': T_ambient,
            'fluid_cp': self.fluid_cp_solar
        }
        collector_out = self.collector.update(dt, collector_inputs)

        # === BUILDING (LOAD) ===
        building_inputs = {
            'T_inlet': self.tank.T_tank,
            'flow_rate': flow_load,
            'T_ambient': T_ambient,
            'irradiance': irradiance,  # For solar gains through windows
            'fluid_cp': self.fluid_cp_load
        }
        building_out = self.building.update(dt, building_inputs)

        # === STORAGE TANK ===
        tank_inputs = {
            'T_inlet_solar': collector_out['T_outlet'],
            'flow_rate_solar': flow_solar,
            'T_inlet_load': building_out['T_outlet'],
            'flow_rate_load': -flow_load,
            'T_ambient': T_ambient,
            'fluid_cp': self.fluid_cp_solar
        }
        tank_out = self.tank.update(dt, tank_inputs)

        # === LOG DATA ===
        self.history['time'].append(time_hours)
        self.history['day_of_year'].append(day_of_year)
        self.history['T_tank'].append(tank_out['T_tank'])
        self.history['T_collector'].append(collector_out['T_collector'])
        self.history['T_building'].append(building_out['T_building'])
        self.history['T_ambient'].append(T_ambient)
        self.history['irradiance_total'].append(solar_data['total'])
        self.history['irradiance_direct'].append(solar_data['direct'])
        self.history['irradiance_diffuse'].append(solar_data['diffuse'])
        self.history['solar_altitude'].append(solar_altitude)
        self.history['cloud_cover'].append(cloud_cover)
        self.history['Q_solar_collected'].append(collector_out['Q_to_fluid'])
        self.history['Q_building_demand'].append(building_out['Q_demand'])
        self.history['Q_building_delivered'].append(building_out['Q_actual'])
        self.history['Q_building_loss'].append(building_out['Q_loss'])
        self.history['Q_building_solar_gain'].append(building_out['Q_solar_gain'])
        self.history['solar_pump_speed'].append(solar_pump_out['speed'])
        self.history['solar_pump_power'].append(solar_pump_out['power'])
        self.history['solar_pump_efficiency'].append(solar_pump_out['efficiency'])
        self.history['load_pump_speed'].append(load_pump_out['speed'])
        self.history['load_pump_power'].append(load_pump_out['power'])
        self.history['load_pump_efficiency'].append(load_pump_out['efficiency'])
        self.history['heating_active'].append(building_out['heating_active'])

    def run_simulation(self, duration_hours: float = 72.0, dt: float = 60.0):
        """Run the complete simulation"""
        num_steps = int(duration_hours * 3600 / dt)

        print("="*70)
        print("REALISTIC SOLAR THERMAL SYSTEM SIMULATION")
        print("="*70)
        print(f"\nLocation: {self.location.latitude}°N, {self.location.longitude}°W")
        print(f"Elevation: {self.location.elevation} m")
        print(f"Start date: Day {self.start_day} of year")
        print(f"Duration: {duration_hours} hours ({duration_hours/24:.1f} days)")
        print(f"\nInitial Conditions:")
        print(f"  Tank temperature: {self.tank.T_tank:.1f}°C")
        print(f"  Building temperature: {self.building.T_building:.1f}°C")
        print(f"  Building setpoint: {self.building.params.setpoint_temp:.1f}°C")
        print(f"\nRunning simulation...")

        for step in range(num_steps):
            time_hours = step * dt / 3600
            self.simulate_timestep(time_hours, dt)

            if step % (num_steps // 10) == 0:
                progress = 100 * step / num_steps
                print(f"  {progress:.0f}% - Tank: {self.tank.T_tank:.1f}°C, "
                      f"Building: {self.building.T_building:.1f}°C")

        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)

        # Results analysis
        print(f"\nFinal Conditions:")
        print(f"  Tank temperature: {self.tank.T_tank:.1f}°C")
        print(f"  Building temperature: {self.building.T_building:.1f}°C")
        print(f"  Max tank temperature: {max(self.history['T_tank']):.1f}°C")

        # Energy analysis
        time_seconds = np.array(self.history['time']) * 3600
        E_solar = np.trapezoid(self.history['Q_solar_collected'], time_seconds) / 1e6
        E_demand = np.trapezoid(self.history['Q_building_demand'], time_seconds) / 1e6
        E_delivered = np.trapezoid(self.history['Q_building_delivered'], time_seconds) / 1e6
        E_loss = np.trapezoid(self.history['Q_building_loss'], time_seconds) / 1e6
        E_solar_gain = np.trapezoid(self.history['Q_building_solar_gain'], time_seconds) / 1e6

        print(f"\nEnergy Summary:")
        print(f"  Solar collected (panels): {E_solar:.1f} MJ")
        print(f"  Building demand: {E_demand:.1f} MJ")
        print(f"  Building delivered: {E_delivered:.1f} MJ")
        print(f"  Solar fraction: {100 * E_delivered / E_demand:.1f}% of demand met by system")
        print(f"  Building heat loss: {E_loss:.1f} MJ")
        print(f"  Passive solar gains: {E_solar_gain:.1f} MJ")

        # Pump energy
        E_pump_solar = np.trapezoid(self.history['solar_pump_power'], time_seconds) / 1e6
        E_pump_load = np.trapezoid(self.history['load_pump_power'], time_seconds) / 1e6
        avg_eta_solar = np.mean([e for e in self.history['solar_pump_efficiency'] if e > 0])
        avg_eta_load = np.mean([e for e in self.history['load_pump_efficiency'] if e > 0])

        print(f"\nPump Performance:")
        print(f"  Solar pump energy: {E_pump_solar:.2f} MJ ({E_pump_solar/E_solar*100:.1f}% of solar collected)")
        print(f"  Load pump energy: {E_pump_load:.2f} MJ")
        print(f"  Solar pump avg efficiency: {avg_eta_solar*100:.1f}%")
        print(f"  Load pump avg efficiency: {avg_eta_load*100:.1f}%")

        # Solar performance
        avg_irradiance = np.mean([i for i in self.history['irradiance_total'] if i > 0])
        avg_cloud = np.mean(self.history['cloud_cover'])

        print(f"\nWeather Conditions:")
        print(f"  Average irradiance (daytime): {avg_irradiance:.0f} W/m²")
        print(f"  Average cloud cover: {avg_cloud*100:.0f}%")
        print(f"  Temperature range: {min(self.history['T_ambient']):.1f} to {max(self.history['T_ambient']):.1f}°C")

        print("="*70)

    def plot_results(self):
        """Create comprehensive results visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        time = np.array(self.history['time'])

        # 1. Solar radiation components
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.fill_between(time, 0, self.history['irradiance_total'],
                         color='yellow', alpha=0.3, label='Total')
        ax1.plot(time, self.history['irradiance_direct'], 'orange',
                linewidth=1.5, label='Direct')
        ax1.plot(time, self.history['irradiance_diffuse'], 'skyblue',
                linewidth=1.5, label='Diffuse')
        ax1.set_ylabel('Irradiance (W/m²)')
        ax1.set_title('Solar Radiation Components')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Sun position and cloud cover
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, self.history['solar_altitude'], 'orange',
                linewidth=2, label='Solar Altitude')
        ax2.set_ylabel('Altitude (degrees)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(time, 0, np.array(self.history['cloud_cover'])*100,
                             color='gray', alpha=0.4, label='Cloud Cover')
        ax2_twin.set_ylabel('Cloud Cover (%)', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        ax2.set_title('Sun Position & Weather')
        ax2.grid(True, alpha=0.3)

        # 3. System temperatures
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time, self.history['T_tank'], 'r-', linewidth=2, label='Tank')
        ax3.plot(time, self.history['T_collector'], 'orange',
                linewidth=2, label='Collector')
        ax3.plot(time, self.history['T_building'], 'b-', linewidth=2, label='Building')
        ax3.plot(time, self.history['T_ambient'], 'gray', linewidth=1.5,
                linestyle='--', label='Ambient')
        ax3.axhline(y=self.building.params.setpoint_temp, color='green',
                   linestyle=':', label='Setpoint')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('System Temperatures')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Solar collection
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time, np.array(self.history['Q_solar_collected'])/1000,
                'g-', linewidth=2)
        ax4.set_ylabel('Heat Power (kW)')
        ax4.set_title('Solar Heat Collection')
        ax4.grid(True, alpha=0.3)

        # 5. Building heat balance
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time, np.array(self.history['Q_building_demand'])/1000,
                'r--', linewidth=2, label='Demand')
        ax5.plot(time, np.array(self.history['Q_building_delivered'])/1000,
                'b-', linewidth=2, label='Delivered')
        ax5.plot(time, np.array(self.history['Q_building_loss'])/1000,
                'orange', linewidth=1.5, alpha=0.7, label='Building Loss')
        ax5.plot(time, np.array(self.history['Q_building_solar_gain'])/1000,
                'yellow', linewidth=1.5, alpha=0.7, label='Passive Solar')
        ax5.set_ylabel('Heat Power (kW)')
        ax5.set_title('Building Heat Balance')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Heating system status
        ax6 = fig.add_subplot(gs[1, 2])
        heating_status = np.array(self.history['heating_active'], dtype=float)
        ax6.fill_between(time, 0, heating_status*100, color='red', alpha=0.3)
        ax6.set_ylabel('Heating Status (%)')
        ax6.set_ylim([-5, 105])
        ax6.set_title('Building Heating Active')
        ax6.grid(True, alpha=0.3)

        # 7. Solar pump performance
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(time, np.array(self.history['solar_pump_speed'])*100,
                'g-', linewidth=2, label='Speed')
        ax7.set_ylabel('Speed (%)', color='green')
        ax7.tick_params(axis='y', labelcolor='green')
        ax7_twin = ax7.twinx()
        ax7_twin.plot(time, np.array(self.history['solar_pump_efficiency'])*100,
                     'b-', linewidth=1.5, alpha=0.7, label='Efficiency')
        ax7_twin.set_ylabel('Efficiency (%)', color='blue')
        ax7_twin.tick_params(axis='y', labelcolor='blue')
        ax7.set_title('Solar Pump Performance')
        ax7.grid(True, alpha=0.3)

        # 8. Load pump performance
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(time, np.array(self.history['load_pump_speed'])*100,
                'b-', linewidth=2, label='Speed')
        ax8.set_ylabel('Speed (%)', color='blue')
        ax8.tick_params(axis='y', labelcolor='blue')
        ax8_twin = ax8.twinx()
        ax8_twin.plot(time, np.array(self.history['load_pump_efficiency'])*100,
                     'g-', linewidth=1.5, alpha=0.7, label='Efficiency')
        ax8_twin.set_ylabel('Efficiency (%)', color='green')
        ax8_twin.tick_params(axis='y', labelcolor='green')
        ax8.set_title('Load Pump Performance')
        ax8.grid(True, alpha=0.3)

        # 9. Pump power consumption
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(time, self.history['solar_pump_power'], 'g-',
                linewidth=2, label='Solar Pump')
        ax9.plot(time, self.history['load_pump_power'], 'b-',
                linewidth=2, label='Load Pump')
        ax9.set_ylabel('Power (W)')
        ax9.set_title('Pump Electrical Power')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        # 10. Cumulative energy
        ax10 = fig.add_subplot(gs[3, :])
        dt_hours = time[1] - time[0] if len(time) > 1 else 1/60
        cum_solar = np.cumsum(self.history['Q_solar_collected']) * dt_hours * 3600 / 1e6
        cum_demand = np.cumsum(self.history['Q_building_demand']) * dt_hours * 3600 / 1e6
        cum_delivered = np.cumsum(self.history['Q_building_delivered']) * dt_hours * 3600 / 1e6
        cum_loss = np.cumsum(self.history['Q_building_loss']) * dt_hours * 3600 / 1e6

        ax10.plot(time, cum_solar, 'g-', linewidth=2.5, label='Solar Collected')
        ax10.plot(time, cum_demand, 'r--', linewidth=2, label='Building Demand')
        ax10.plot(time, cum_delivered, 'b-', linewidth=2, label='Heating Delivered')
        ax10.plot(time, cum_loss, 'orange', linewidth=1.5, alpha=0.7, label='Building Loss')
        ax10.set_xlabel('Time (hours)')
        ax10.set_ylabel('Cumulative Energy (MJ)')
        ax10.set_title('Cumulative Energy Balance')
        ax10.legend(loc='best')
        ax10.grid(True, alpha=0.3)

        fig.suptitle('Realistic Solar Thermal System - Comprehensive Results',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.savefig('realistic_system_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'realistic_system_results.png'")
        plt.show()


def main():
    """Run realistic simulation"""

    # Location (Denver, CO as example)
    location = LocationParams(
        latitude=39.7,
        longitude=-105.0,
        timezone=-7.0,
        elevation=1600.0
    )

    # Solar collector
    collector = SolarCollector(
        "SolarCollector",
        SolarCollectorParams(
            area=6.0,  # Larger array for commercial building
            efficiency=0.75,
            heat_loss_coef=4.5
        )
    )

    # Storage tank
    tank = StorageTank(
        "StorageTank",
        StorageTankParams(
            volume=0.5,  # 500 liters
            mass=500.0,
            surface_area=2.5
        ),
        initial_temp=40.0
    )

    # Building with thermal mass
    building = BuildingThermalMass(
        "CommercialBuilding",
        BuildingParams(
            floor_area=1000.0,  # 1000 m² building
            thermal_mass=75000.0,  # Heavy construction
            UA_envelope=600.0,  # W/K
            internal_gains=3000.0,  # Significant occupancy
            setpoint_temp=21.0,
            solar_aperture=80.0  # Large windows
        ),
        initial_temp=20.0
    )

    # Controller
    controller = BasicController(
        solar_dT_threshold=8.0,
        tank_max_temp=80.0
    )

    # Create system
    system = RealisticThermalSystem(
        collector=collector,
        tank=tank,
        building=building,
        controller=controller,
        location=location,
        start_day=180,  # Summer solstice region
        ambient_temp=22.0
    )

    # Run 72-hour simulation
    system.run_simulation(duration_hours=72.0, dt=60.0)

    # Generate plots
    system.plot_results()


if __name__ == "__main__":
    main()
