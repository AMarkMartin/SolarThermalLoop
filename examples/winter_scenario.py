"""
Winter Heating Scenario
- Solar provides portion of building heating demand
- Auxiliary heating (natural gas/electric) covers remaining load
- Simplified building model for realistic demand signal
- Focus on solar loop physics fidelity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    Component
)
from src.models import (
    SolarRadiationModel, LocationParams, WeatherForecast,
    PumpWithCurve
)
from src.control import BasicController


# ============================================================================
# SIMPLIFIED BUILDING LOAD MODEL
# ============================================================================

class SimplifiedBuildingLoad(Component):
    """
    Simplified building model that generates realistic heating demand
    correlated with weather. Auxiliary heating handles any shortfall.
    """

    def __init__(
        self,
        name: str,
        floor_area: float = 1000.0,
        design_heat_load: float = 15000.0,  # W at design conditions
        design_outdoor_temp: float = -10.0,  # °C
        indoor_setpoint: float = 21.0  # °C
    ):
        super().__init__(name)
        self.floor_area = floor_area
        self.design_heat_load = design_heat_load
        self.design_outdoor_temp = design_outdoor_temp
        self.indoor_setpoint = indoor_setpoint

        # Calculate building UA from design conditions
        # Q = UA * (T_in - T_out)
        self.UA = design_heat_load / (indoor_setpoint - design_outdoor_temp)

        # Current state
        self.Q_total_demand = 0.0
        self.Q_solar_delivered = 0.0
        self.Q_auxiliary = 0.0
        self.T_return = indoor_setpoint - 10.0  # Typical return temp

    def calculate_heating_demand(self, T_outdoor: float, hour: int) -> float:
        """
        Calculate building heating demand based on outdoor temperature.
        Includes night setback and occupancy effects.
        """
        # Base load from envelope
        Q_envelope = self.UA * (self.indoor_setpoint - T_outdoor)

        # Occupancy/usage pattern (higher during occupied hours)
        hour_of_day = hour % 24
        if 6 <= hour_of_day <= 22:  # Occupied
            occupancy_factor = 1.0
        else:  # Night setback
            occupancy_factor = 0.7

        # Total heating demand
        Q_demand = max(Q_envelope * occupancy_factor, 0.0)

        return Q_demand

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update building heating demand and calculate auxiliary requirement.

        Inputs:
            - T_supply: Supply temp from solar system (°C)
            - flow_rate: Flow rate from solar system (kg/s)
            - T_outdoor: Outdoor temperature (°C)
            - hour: Current hour (for occupancy pattern)
        """
        T_supply = inputs.get('T_supply', 30.0)
        flow_rate = inputs.get('flow_rate', 0.0)
        T_outdoor = inputs.get('T_outdoor', 0.0)
        hour = inputs.get('hour', 12)
        fluid_cp = inputs.get('fluid_cp', 4186.0)

        # Calculate total heating demand
        self.Q_total_demand = self.calculate_heating_demand(T_outdoor, hour)

        # Heat delivered by solar system
        # Limited by available temperature and flow
        if flow_rate > 0 and T_supply > self.T_return:
            # Assume building heat exchanger extracts heat
            dT_max = T_supply - self.T_return
            Q_available = flow_rate * fluid_cp * dT_max

            # Deliver up to the demand, limited by availability
            self.Q_solar_delivered = min(Q_available, self.Q_total_demand)

            # Calculate actual return temperature
            dT_actual = self.Q_solar_delivered / (flow_rate * fluid_cp)
            self.T_return = T_supply - dT_actual
        else:
            self.Q_solar_delivered = 0.0
            self.T_return = self.indoor_setpoint - 10.0

        # Auxiliary heating makes up the difference
        self.Q_auxiliary = max(self.Q_total_demand - self.Q_solar_delivered, 0.0)

        # Solar fraction
        solar_fraction = (self.Q_solar_delivered / self.Q_total_demand
                         if self.Q_total_demand > 0 else 0.0)

        return {
            'Q_total_demand': self.Q_total_demand,
            'Q_solar_delivered': self.Q_solar_delivered,
            'Q_auxiliary': self.Q_auxiliary,
            'solar_fraction': solar_fraction,
            'T_return': self.T_return
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'Q_demand': self.Q_total_demand,
            'Q_solar': self.Q_solar_delivered,
            'Q_aux': self.Q_auxiliary,
            'UA': self.UA
        }


# ============================================================================
# WINTER SCENARIO SYSTEM
# ============================================================================

class WinterHeatingSystem:
    """
    Winter heating scenario with solar providing portion of load.
    Focus on solar loop physics with simplified but realistic building demand.
    """

    def __init__(
        self,
        collector: SolarCollector,
        tank: StorageTank,
        building: SimplifiedBuildingLoad,
        controller: BasicController,
        location: LocationParams,
        start_day: int = 355,  # December
        base_temp: float = -5.0  # Winter baseline
    ):
        self.collector = collector
        self.tank = tank
        self.building = building
        self.controller = controller

        # Pumps with curves
        self.solar_pump = PumpWithCurve(
            "SolarPump",
            rated_flow=0.03,
            rated_head=4.0,
            max_head=6.0,
            efficiency_bep=0.68
        )
        self.load_pump = PumpWithCurve(
            "LoadPump",
            rated_flow=0.08,  # Larger for heating loads
            rated_head=5.0,
            max_head=7.5,
            efficiency_bep=0.72
        )

        # Environment
        self.location = location
        self.solar_model = SolarRadiationModel(location)
        self.weather = WeatherForecast(days=5)
        self.start_day = start_day
        self.base_temp = base_temp

        # Fluid properties
        self.fluid_cp_solar = 3500.0  # Glycol
        self.fluid_cp_load = 4186.0  # Water

        # Cost parameters
        self.gas_price = 0.04  # $/kWh (natural gas)
        self.elec_price = 0.12  # $/kWh (electricity)

        # History
        self.history: Dict[str, List] = {
            'time': [],
            'day_of_year': [],
            'hour_of_day': [],
            'T_tank': [],
            'T_collector': [],
            'T_outdoor': [],
            'T_supply': [],
            'T_return': [],
            'irradiance_total': [],
            'irradiance_direct': [],
            'irradiance_diffuse': [],
            'solar_altitude': [],
            'cloud_cover': [],
            'Q_solar_collected': [],
            'Q_building_demand': [],
            'Q_solar_delivered': [],
            'Q_auxiliary': [],
            'solar_fraction': [],
            'solar_pump_speed': [],
            'solar_pump_power': [],
            'solar_pump_efficiency': [],
            'load_pump_speed': [],
            'load_pump_power': [],
            'load_pump_efficiency': [],
            'aux_cost': [],
        }

    def get_outdoor_temperature(self, time_hours: float) -> float:
        """Winter temperature with realistic variation"""
        hour = int(time_hours)
        day = int(time_hours / 24)

        # Get forecast temperature
        T_forecast = self.weather.get_temperature(hour)

        # Winter pattern: colder at night
        hour_of_day = time_hours % 24
        T_diurnal = self.base_temp - 6 * np.cos(2 * np.pi * (hour_of_day - 15) / 24)

        # Add some day-to-day variation
        T_daily_var = 3 * np.sin(2 * np.pi * day / 7)

        # Blend
        T_outdoor = 0.5 * T_forecast + 0.3 * T_diurnal + 0.2 * T_daily_var

        return T_outdoor

    def simulate_timestep(self, time_hours: float, dt: float):
        """Simulate one timestep"""

        day_of_year = self.start_day + int(time_hours / 24)
        hour_of_day = time_hours % 24
        hour = int(time_hours)

        # Weather
        T_outdoor = self.get_outdoor_temperature(time_hours)
        cloud_cover = self.weather.get_cloud_cover(hour)

        # Solar radiation
        solar_data = self.solar_model.calculate_irradiance(
            time_hours=hour_of_day,
            day_of_year=day_of_year,
            cloud_cover=cloud_cover,
            panel_tilt=50.0,  # Steeper for winter (latitude + 10°)
            panel_azimuth=180.0
        )

        irradiance = solar_data['total']

        # === CONTROL ===
        system_state = {
            'T_collector': self.collector.T_collector,
            'T_tank': self.tank.T_tank,
            'Q_demand': self.building.Q_total_demand,
            'irradiance': irradiance,
            'time_hours': time_hours
        }

        control = self.controller.compute_control(system_state)

        # === PUMPS ===
        solar_pump_out = self.solar_pump.update(dt, {'speed': control['solar_pump_speed']})
        load_pump_out = self.load_pump.update(dt, {'speed': control['load_pump_speed']})

        flow_solar = solar_pump_out['flow_rate']
        flow_load = load_pump_out['flow_rate']

        # === SOLAR COLLECTOR ===
        collector_inputs = {
            'T_inlet': self.tank.T_tank,
            'flow_rate': flow_solar,
            'irradiance': irradiance,
            'T_ambient': T_outdoor,
            'fluid_cp': self.fluid_cp_solar
        }
        collector_out = self.collector.update(dt, collector_inputs)

        # === BUILDING LOAD ===
        building_inputs = {
            'T_supply': self.tank.T_tank,
            'flow_rate': flow_load,
            'T_outdoor': T_outdoor,
            'hour': hour,
            'fluid_cp': self.fluid_cp_load
        }
        building_out = self.building.update(dt, building_inputs)

        # === STORAGE TANK ===
        tank_inputs = {
            'T_inlet_solar': collector_out['T_outlet'],
            'flow_rate_solar': flow_solar,
            'T_inlet_load': building_out['T_return'],
            'flow_rate_load': flow_load,  # Positive - tank calculates heat extraction correctly
            'T_ambient': T_outdoor,
            'fluid_cp': self.fluid_cp_solar
        }
        tank_out = self.tank.update(dt, tank_inputs)

        # Calculate auxiliary heating cost
        aux_cost_step = (building_out['Q_auxiliary'] * dt / 3.6e6) * self.gas_price  # $

        # === LOG ===
        self.history['time'].append(time_hours)
        self.history['day_of_year'].append(day_of_year)
        self.history['hour_of_day'].append(hour_of_day)
        self.history['T_tank'].append(tank_out['T_tank'])
        self.history['T_collector'].append(collector_out['T_collector'])
        self.history['T_outdoor'].append(T_outdoor)
        self.history['T_supply'].append(self.tank.T_tank)
        self.history['T_return'].append(building_out['T_return'])
        self.history['irradiance_total'].append(solar_data['total'])
        self.history['irradiance_direct'].append(solar_data['direct'])
        self.history['irradiance_diffuse'].append(solar_data['diffuse'])
        self.history['solar_altitude'].append(solar_data['altitude'])
        self.history['cloud_cover'].append(cloud_cover)
        self.history['Q_solar_collected'].append(collector_out['Q_to_fluid'])
        self.history['Q_building_demand'].append(building_out['Q_total_demand'])
        self.history['Q_solar_delivered'].append(building_out['Q_solar_delivered'])
        self.history['Q_auxiliary'].append(building_out['Q_auxiliary'])
        self.history['solar_fraction'].append(building_out['solar_fraction'])
        self.history['solar_pump_speed'].append(solar_pump_out['speed'])
        self.history['solar_pump_power'].append(solar_pump_out['power'])
        self.history['solar_pump_efficiency'].append(solar_pump_out['efficiency'])
        self.history['load_pump_speed'].append(load_pump_out['speed'])
        self.history['load_pump_power'].append(load_pump_out['power'])
        self.history['load_pump_efficiency'].append(load_pump_out['efficiency'])
        self.history['aux_cost'].append(aux_cost_step)

    def run_simulation(self, duration_hours: float = 120.0, dt: float = 60.0):
        """Run winter simulation"""
        num_steps = int(duration_hours * 3600 / dt)

        print("="*70)
        print("WINTER HEATING SCENARIO - SOLAR SUPPLEMENTAL HEATING")
        print("="*70)
        print(f"\nLocation: {self.location.latitude}°N, {self.location.longitude}°W")
        print(f"Season: Winter (Day {self.start_day} of year)")
        print(f"Duration: {duration_hours/24:.1f} days")
        print(f"\nSystem Sizing:")
        print(f"  Solar collector: {self.collector.params.area} m²")
        print(f"  Storage tank: {self.tank.params.volume * 1000:.0f} liters")
        print(f"  Building: {self.building.floor_area} m²")
        print(f"  Design heat load: {self.building.design_heat_load/1000:.1f} kW at {self.building.design_outdoor_temp}°C")
        print(f"\nInitial Conditions:")
        print(f"  Tank: {self.tank.T_tank:.1f}°C")
        print(f"  Outdoor: {self.base_temp:.1f}°C baseline")
        print(f"\nRunning simulation...")

        for step in range(num_steps):
            time_hours = step * dt / 3600
            self.simulate_timestep(time_hours, dt)

            if step % (num_steps // 10) == 0:
                progress = 100 * step / num_steps
                print(f"  {progress:.0f}% - Tank: {self.tank.T_tank:.1f}°C, "
                      f"Outdoor: {self.history['T_outdoor'][-1]:.1f}°C, "
                      f"Demand: {self.history['Q_building_demand'][-1]/1000:.1f} kW")

        print("\n" + "="*70)
        print("WINTER SCENARIO RESULTS")
        print("="*70)

        # Energy analysis
        time_seconds = np.array(self.history['time']) * 3600
        E_solar_collected = np.trapezoid(self.history['Q_solar_collected'], time_seconds) / 1e6
        E_demand = np.trapezoid(self.history['Q_building_demand'], time_seconds) / 1e6
        E_solar_delivered = np.trapezoid(self.history['Q_solar_delivered'], time_seconds) / 1e6
        E_auxiliary = np.trapezoid(self.history['Q_auxiliary'], time_seconds) / 1e6

        avg_solar_fraction = np.mean([sf for sf in self.history['solar_fraction']])

        print(f"\nEnergy Summary:")
        print(f"  Building demand: {E_demand:.1f} MJ ({E_demand/3.6:.1f} kWh)")
        print(f"  Solar collected: {E_solar_collected:.1f} MJ ({E_solar_collected/3.6:.1f} kWh)")
        print(f"  Solar delivered: {E_solar_delivered:.1f} MJ ({E_solar_delivered/3.6:.1f} kWh)")
        print(f"  Auxiliary heating: {E_auxiliary:.1f} MJ ({E_auxiliary/3.6:.1f} kWh)")
        print(f"  Average solar fraction: {avg_solar_fraction*100:.1f}%")

        # Economics
        total_aux_cost = sum(self.history['aux_cost'])
        E_pump = (np.trapezoid(self.history['solar_pump_power'], time_seconds) +
                 np.trapezoid(self.history['load_pump_power'], time_seconds)) / 3.6e6
        pump_cost = E_pump * self.elec_price

        print(f"\nEconomics ({duration_hours/24:.0f} days):")
        print(f"  Auxiliary fuel cost: ${total_aux_cost:.2f}")
        print(f"  Pump electricity cost: ${pump_cost:.2f}")
        print(f"  Total operating cost: ${total_aux_cost + pump_cost:.2f}")
        print(f"  Solar savings: ${(E_solar_delivered/3.6) * self.gas_price:.2f}")

        # Solar performance
        avg_irradiance = np.mean([i for i in self.history['irradiance_total'] if i > 0])
        max_altitude = max(self.history['solar_altitude'])

        print(f"\nSolar Performance:")
        print(f"  Avg daytime irradiance: {avg_irradiance:.0f} W/m²")
        print(f"  Max sun altitude: {max_altitude:.1f}° (winter low sun)")
        print(f"  Avg cloud cover: {np.mean(self.history['cloud_cover'])*100:.0f}%")
        print(f"  Collector efficiency: {(E_solar_collected/(E_demand))*100:.1f}% of demand")

        # Temperature stats
        print(f"\nTemperature Stats:")
        print(f"  Outdoor range: {min(self.history['T_outdoor']):.1f} to {max(self.history['T_outdoor']):.1f}°C")
        print(f"  Tank range: {min(self.history['T_tank']):.1f} to {max(self.history['T_tank']):.1f}°C")
        print(f"  Max collector: {max(self.history['T_collector']):.1f}°C")

        print("="*70)

    def plot_results(self):
        """Comprehensive winter scenario plots"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        time = np.array(self.history['time'])

        # 1. Outdoor temperature and solar radiation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, self.history['T_outdoor'], 'b-', linewidth=2, label='Outdoor Temp')
        ax1.set_ylabel('Temperature (°C)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(time, 0, self.history['irradiance_total'],
                              color='yellow', alpha=0.4, label='Irradiance')
        ax1_twin.set_ylabel('Irradiance (W/m²)', color='orange')
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        ax1.set_title('Weather Conditions')
        ax1.grid(True, alpha=0.3)

        # 2. System temperatures
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, self.history['T_tank'], 'r-', linewidth=2.5, label='Tank')
        ax2.plot(time, self.history['T_collector'], 'orange', linewidth=2, label='Collector')
        ax2.plot(time, self.history['T_supply'], 'm-', linewidth=1.5, alpha=0.7, label='Supply')
        ax2.plot(time, self.history['T_return'], 'c-', linewidth=1.5, alpha=0.7, label='Return')
        ax2.axhline(y=40, color='gray', linestyle='--', linewidth=1, label='Useful Temp')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('System Temperatures')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Solar collection
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time, np.array(self.history['Q_solar_collected'])/1000,
                'g-', linewidth=2)
        ax3.fill_between(time, 0, np.array(self.history['Q_solar_collected'])/1000,
                        color='green', alpha=0.3)
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Solar Heat Collection')
        ax3.grid(True, alpha=0.3)

        # 4. Building heating loads
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time, np.array(self.history['Q_building_demand'])/1000,
                'k--', linewidth=2, label='Total Demand')
        ax4.plot(time, np.array(self.history['Q_solar_delivered'])/1000,
                'g-', linewidth=2, label='Solar')
        ax4.plot(time, np.array(self.history['Q_auxiliary'])/1000,
                'r-', linewidth=2, label='Auxiliary')
        ax4.fill_between(time, 0, np.array(self.history['Q_solar_delivered'])/1000,
                        color='green', alpha=0.2)
        ax4.fill_between(time, np.array(self.history['Q_solar_delivered'])/1000,
                        np.array(self.history['Q_building_demand'])/1000,
                        color='red', alpha=0.2)
        ax4.set_ylabel('Power (kW)')
        ax4.set_title('Building Heating Breakdown')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. Solar fraction
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time, np.array(self.history['solar_fraction'])*100,
                'g-', linewidth=2)
        ax5.fill_between(time, 0, np.array(self.history['solar_fraction'])*100,
                        color='green', alpha=0.3)
        ax5.axhline(y=np.mean(self.history['solar_fraction'])*100,
                   color='blue', linestyle='--', linewidth=2,
                   label=f'Avg: {np.mean(self.history['solar_fraction'])*100:.1f}%')
        ax5.set_ylabel('Solar Fraction (%)')
        ax5.set_ylim([0, 105])
        ax5.set_title('Instantaneous Solar Fraction')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Pump operation
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time, np.array(self.history['solar_pump_speed'])*100,
                'g-', linewidth=2, label='Solar Pump')
        ax6.plot(time, np.array(self.history['load_pump_speed'])*100,
                'b-', linewidth=2, label='Load Pump')
        ax6.set_ylabel('Pump Speed (%)')
        ax6.set_title('Pump Control')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([-5, 105])

        # 7. Cumulative energy
        ax7 = fig.add_subplot(gs[2, :2])
        dt_hours = time[1] - time[0] if len(time) > 1 else 1/60
        cum_demand = np.cumsum(self.history['Q_building_demand']) * dt_hours * 3600 / 1e6
        cum_solar = np.cumsum(self.history['Q_solar_delivered']) * dt_hours * 3600 / 1e6
        cum_aux = np.cumsum(self.history['Q_auxiliary']) * dt_hours * 3600 / 1e6

        ax7.plot(time, cum_demand, 'k--', linewidth=2.5, label='Total Demand')
        ax7.plot(time, cum_solar, 'g-', linewidth=2.5, label='Solar Contribution')
        ax7.plot(time, cum_aux, 'r-', linewidth=2, label='Auxiliary')
        ax7.fill_between(time, 0, cum_solar, color='green', alpha=0.2)
        ax7.fill_between(time, cum_solar, cum_demand, color='red', alpha=0.2)
        ax7.set_xlabel('Time (hours)')
        ax7.set_ylabel('Cumulative Energy (MJ)')
        ax7.set_title('Cumulative Energy Balance')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Auxiliary cost accumulation
        ax8 = fig.add_subplot(gs[2, 2])
        cum_cost = np.cumsum(self.history['aux_cost'])
        ax8.plot(time, cum_cost, 'r-', linewidth=2.5)
        ax8.fill_between(time, 0, cum_cost, color='red', alpha=0.2)
        ax8.set_xlabel('Time (hours)')
        ax8.set_ylabel('Cost ($)')
        ax8.set_title(f'Auxiliary Heating Cost (${cum_cost[-1]:.2f} total)')
        ax8.grid(True, alpha=0.3)

        # 9. Solar altitude and cloud
        ax9 = fig.add_subplot(gs[3, 0])
        ax9.plot(time, self.history['solar_altitude'], 'orange', linewidth=2)
        ax9.fill_between(time, 0, self.history['solar_altitude'],
                        where=np.array(self.history['solar_altitude']) > 0,
                        color='yellow', alpha=0.3)
        ax9.set_ylabel('Solar Altitude (°)', color='orange')
        ax9.tick_params(axis='y', labelcolor='orange')
        ax9_twin = ax9.twinx()
        ax9_twin.fill_between(time, 0, np.array(self.history['cloud_cover'])*100,
                             color='gray', alpha=0.4)
        ax9_twin.set_ylabel('Cloud Cover (%)', color='gray')
        ax9_twin.tick_params(axis='y', labelcolor='gray')
        ax9.set_xlabel('Time (hours)')
        ax9.set_title('Sun Position & Clouds')
        ax9.grid(True, alpha=0.3)

        # 10. Pump efficiency
        ax10 = fig.add_subplot(gs[3, 1])
        ax10.plot(time, np.array(self.history['solar_pump_efficiency'])*100,
                 'g-', linewidth=1.5, alpha=0.7, label='Solar Pump')
        ax10.plot(time, np.array(self.history['load_pump_efficiency'])*100,
                 'b-', linewidth=1.5, alpha=0.7, label='Load Pump')
        ax10.set_xlabel('Time (hours)')
        ax10.set_ylabel('Efficiency (%)')
        ax10.set_title('Pump Efficiency')
        ax10.legend()
        ax10.grid(True, alpha=0.3)

        # 11. Pump power
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.plot(time, self.history['solar_pump_power'], 'g-',
                 linewidth=2, label='Solar')
        ax11.plot(time, self.history['load_pump_power'], 'b-',
                 linewidth=2, label='Load')
        ax11.set_xlabel('Time (hours)')
        ax11.set_ylabel('Power (W)')
        ax11.set_title('Pump Electrical Power')
        ax11.legend()
        ax11.grid(True, alpha=0.3)

        fig.suptitle('Winter Heating Scenario - Solar Supplemental System',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.savefig('winter_scenario_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'winter_scenario_results.png'")
        plt.show()


def main():
    """Run shoulder-season heating scenario"""

    # Location
    location = LocationParams(
        latitude=40.0,
        longitude=-105.0,
        timezone=-7.0,
        elevation=1600.0
    )

    # Sized for ~25-30% solar fraction in shoulder season
    collector = SolarCollector(
        "SolarCollector",
        SolarCollectorParams(
            area=15.0,  # Residential-scale system
            efficiency=0.75,
            heat_loss_coef=4.0
        )
    )

    # Storage sized for ~1 day of solar collection
    tank = StorageTank(
        "StorageTank",
        StorageTankParams(
            volume=0.8,  # 800 liters
            mass=800.0,
            surface_area=4.5,
            heat_loss_coef=1.0  # Well-insulated
        ),
        initial_temp=35.0
    )

    # Residential building load
    building = SimplifiedBuildingLoad(
        "ResidentialBuilding",
        floor_area=200.0,
        design_heat_load=5000.0,  # 5 kW at design
        design_outdoor_temp=-10.0,
        indoor_setpoint=21.0
    )

    # Controller
    controller = BasicController(
        solar_dT_threshold=5.0,
        tank_max_temp=75.0
    )

    # Create system
    system = WinterHeatingSystem(
        collector=collector,
        tank=tank,
        building=building,
        controller=controller,
        location=location,
        start_day=75,   # Early March — shoulder season
        base_temp=5.0
    )

    # Run 5-day simulation
    system.run_simulation(duration_hours=120.0, dt=60.0)

    # Plot
    system.plot_results()


if __name__ == "__main__":
    main()
