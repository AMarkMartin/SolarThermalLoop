"""
Enhanced physics models for realistic simulation:
- Realistic solar radiation with sun position calculations
- Weather forecasting capability
- Pump performance curves
- Building thermal mass model
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from components import Component, FluidComponent, ThermalLoad


# ============================================================================
# REALISTIC SOLAR RADIATION MODEL
# ============================================================================

@dataclass
class LocationParams:
    """Geographic location parameters"""
    latitude: float = 40.0  # degrees North (e.g., Denver, CO)
    longitude: float = -105.0  # degrees West
    timezone: float = -7.0  # UTC offset
    elevation: float = 1600.0  # meters (Denver elevation)


class SolarRadiationModel:
    """
    Realistic solar radiation model with:
    - Sun position calculations (solar altitude and azimuth)
    - Clear sky radiation model
    - Cloud cover effects
    - Atmospheric attenuation
    """

    def __init__(self, location: LocationParams):
        self.location = location
        self.solar_constant = 1367.0  # W/m² - extraterrestrial solar radiation

    def solar_declination(self, day_of_year: int) -> float:
        """Calculate solar declination angle (degrees)"""
        # Cooper's equation
        delta = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        return delta

    def equation_of_time(self, day_of_year: int) -> float:
        """Calculate equation of time (minutes)"""
        B = 360 * (day_of_year - 81) / 364
        B_rad = np.radians(B)
        EoT = 9.87 * np.sin(2 * B_rad) - 7.53 * np.cos(B_rad) - 1.5 * np.sin(B_rad)
        return EoT

    def solar_time(self, clock_time: float, day_of_year: int) -> float:
        """Convert clock time to solar time"""
        EoT = self.equation_of_time(day_of_year)
        # LSTM = Local Standard Time Meridian
        LSTM = 15 * self.location.timezone
        TC = 4 * (self.location.longitude - LSTM) + EoT
        solar_time = clock_time + TC / 60
        return solar_time

    def solar_position(self, time_hours: float, day_of_year: int) -> tuple:
        """
        Calculate solar altitude and azimuth angles.

        Returns:
            (altitude, azimuth) in degrees
        """
        # Solar time
        solar_t = self.solar_time(time_hours, day_of_year)

        # Hour angle (degrees)
        hour_angle = 15 * (solar_t - 12)

        # Declination
        delta = self.solar_declination(day_of_year)

        # Latitude
        lat = self.location.latitude

        # Solar altitude angle (elevation)
        sin_altitude = (np.sin(np.radians(lat)) * np.sin(np.radians(delta)) +
                       np.cos(np.radians(lat)) * np.cos(np.radians(delta)) *
                       np.cos(np.radians(hour_angle)))
        altitude = np.degrees(np.arcsin(np.clip(sin_altitude, -1, 1)))

        # Solar azimuth angle
        cos_azimuth = ((np.sin(np.radians(delta)) * np.cos(np.radians(lat)) -
                       np.cos(np.radians(delta)) * np.sin(np.radians(lat)) *
                       np.cos(np.radians(hour_angle))) /
                      np.cos(np.radians(altitude)))
        azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))

        if hour_angle > 0:
            azimuth = 360 - azimuth

        return altitude, azimuth

    def clear_sky_radiation(self, altitude: float) -> float:
        """
        Calculate clear sky direct normal irradiance using simplified model.

        Args:
            altitude: Solar altitude angle (degrees)

        Returns:
            Direct normal irradiance (W/m²)
        """
        if altitude <= 0:
            return 0.0

        # Air mass
        AM = 1 / (np.sin(np.radians(altitude)) + 0.50572 * (altitude + 6.07995)**-1.6364)

        # Atmospheric attenuation (simplified)
        # Account for elevation
        pressure_ratio = np.exp(-self.location.elevation / 8400)  # Scale height ~8.4 km
        tau = 0.7**AM * pressure_ratio

        # Direct normal irradiance
        DNI = self.solar_constant * tau

        return DNI

    def calculate_irradiance(
        self,
        time_hours: float,
        day_of_year: int,
        cloud_cover: float = 0.0,
        panel_tilt: float = 40.0,
        panel_azimuth: float = 180.0
    ) -> Dict[str, float]:
        """
        Calculate total irradiance on a tilted surface.

        Args:
            time_hours: Time of day (0-24)
            day_of_year: Day of year (1-365)
            cloud_cover: Fraction (0-1, 0=clear, 1=overcast)
            panel_tilt: Panel tilt from horizontal (degrees)
            panel_azimuth: Panel azimuth (degrees, 180=south)

        Returns:
            Dictionary with irradiance components
        """
        # Sun position
        altitude, azimuth = self.solar_position(time_hours, day_of_year)

        if altitude <= 0:
            return {
                'total': 0.0,
                'direct': 0.0,
                'diffuse': 0.0,
                'altitude': altitude,
                'azimuth': azimuth
            }

        # Clear sky DNI
        DNI = self.clear_sky_radiation(altitude)

        # Direct component on tilted surface
        # Incidence angle
        cos_incidence = (np.sin(np.radians(altitude)) * np.cos(np.radians(panel_tilt)) +
                        np.cos(np.radians(altitude)) * np.sin(np.radians(panel_tilt)) *
                        np.cos(np.radians(azimuth - panel_azimuth)))
        cos_incidence = max(cos_incidence, 0)

        direct = DNI * cos_incidence * (1 - cloud_cover)

        # Diffuse component (simplified sky model)
        DHI = 0.15 * DNI  # Diffuse horizontal irradiance (clear sky)
        diffuse = DHI * (1 + np.cos(np.radians(panel_tilt))) / 2
        diffuse *= (1 + cloud_cover)  # More diffuse when cloudy

        # Ground reflected (albedo = 0.2)
        GHI = DNI * np.sin(np.radians(altitude)) + DHI
        ground_reflected = GHI * 0.2 * (1 - np.cos(np.radians(panel_tilt))) / 2

        total = direct + diffuse + ground_reflected

        return {
            'total': total,
            'direct': direct,
            'diffuse': diffuse,
            'ground': ground_reflected,
            'altitude': altitude,
            'azimuth': azimuth,
            'DNI': DNI
        }


class WeatherForecast:
    """Simple weather forecast generator"""

    def __init__(self, days: int = 3):
        self.days = days
        self.cloud_forecast: List[float] = []
        self.temp_forecast: List[float] = []
        self._generate_forecast()

    def _generate_forecast(self):
        """Generate synthetic weather forecast"""
        hours = self.days * 24
        # Realistic cloud cover pattern (0-1)
        # Mix of clear and partly cloudy days
        for day in range(self.days):
            if np.random.random() < 0.7:  # 70% chance of good weather
                daily_cloud = 0.1 + 0.2 * np.random.random()  # Mostly clear
            else:
                daily_cloud = 0.4 + 0.4 * np.random.random()  # Partly cloudy

            # Diurnal variation
            for hour in range(24):
                variation = 0.1 * np.sin(2 * np.pi * hour / 24)
                self.cloud_forecast.append(np.clip(daily_cloud + variation, 0, 1))

        # Temperature forecast (°C)
        for day in range(self.days):
            T_min = 15 + 5 * np.random.randn()
            T_max = T_min + 10 + 3 * np.random.randn()
            for hour in range(24):
                # Sinusoidal daily temperature
                T_avg = (T_min + T_max) / 2
                T_amp = (T_max - T_min) / 2
                T = T_avg - T_amp * np.cos(2 * np.pi * (hour - 15) / 24)
                self.temp_forecast.append(T)

    def get_cloud_cover(self, hour: int) -> float:
        """Get forecasted cloud cover for given hour"""
        if 0 <= hour < len(self.cloud_forecast):
            return self.cloud_forecast[hour]
        return 0.2  # Default

    def get_temperature(self, hour: int) -> float:
        """Get forecasted temperature for given hour"""
        if 0 <= hour < len(self.temp_forecast):
            return self.temp_forecast[hour]
        return 20.0  # Default


# ============================================================================
# PUMP PERFORMANCE CURVES
# ============================================================================

class PumpWithCurve(Component):
    """
    Pump with realistic performance curve.
    Head and efficiency vary with flow rate.
    """

    def __init__(
        self,
        name: str,
        rated_flow: float = 0.05,  # kg/s at BEP
        rated_head: float = 5.0,  # meters at BEP
        max_head: float = 8.0,  # shutoff head (m)
        efficiency_bep: float = 0.70  # efficiency at best efficiency point
    ):
        super().__init__(name)
        self.rated_flow = rated_flow
        self.rated_head = rated_head
        self.max_head = max_head
        self.efficiency_bep = efficiency_bep

        self.speed = 1.0  # Speed setpoint (0-1)
        self.flow_rate = 0.0
        self.head = 0.0
        self.efficiency = 0.0
        self.power = 0.0

    def pump_curve(self, flow_fraction: float) -> float:
        """
        Calculate head as function of flow (quadratic curve).

        Args:
            flow_fraction: Flow as fraction of rated (0-1.5)

        Returns:
            Head in meters
        """
        # Quadratic pump curve: H = H_max - k * Q²
        # At rated flow: H = rated_head
        k = (self.max_head - self.rated_head) / (self.rated_flow**2)
        Q = flow_fraction * self.rated_flow * self.speed

        head = self.max_head * self.speed**2 - k * Q**2
        return max(head, 0.0)

    def efficiency_curve(self, flow_fraction: float) -> float:
        """
        Calculate efficiency as function of flow.
        Peak at BEP, lower at partial/excessive flow.
        """
        # Parabolic efficiency curve
        # Peak at flow_fraction = 1.0
        eta = self.efficiency_bep * (1 - 0.7 * (flow_fraction - 1.0)**2)
        return np.clip(eta, 0.1, self.efficiency_bep)

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update pump operation with performance curves.

        Inputs:
            - speed: Commanded speed (0-1)
            - system_resistance: Optional system resistance (Pa·s/kg)
        """
        self.speed = np.clip(inputs.get('speed', self.speed), 0.0, 1.0)

        if self.speed < 0.01:
            self.flow_rate = 0.0
            self.head = 0.0
            self.efficiency = 0.0
            self.power = 0.0
        else:
            # For now, assume flow proportional to speed (simplified)
            # In reality, would solve pump curve vs system curve
            flow_fraction = self.speed
            self.flow_rate = flow_fraction * self.rated_flow

            # Calculate head and efficiency
            self.head = self.pump_curve(flow_fraction)
            self.efficiency = self.efficiency_curve(flow_fraction)

            # Power consumption (W)
            # P = ρ * g * Q * H / η
            rho = 1000  # kg/m³
            g = 9.81  # m/s²
            Q_m3s = self.flow_rate / rho
            self.power = (rho * g * Q_m3s * self.head / self.efficiency) if self.efficiency > 0 else 0

        return {
            'flow_rate': self.flow_rate,
            'speed': self.speed,
            'head': self.head,
            'efficiency': self.efficiency,
            'power': self.power  # Watts
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'speed': self.speed,
            'flow_rate': self.flow_rate,
            'head': self.head,
            'efficiency': self.efficiency,
            'power': self.power
        }


# ============================================================================
# BUILDING THERMAL MASS MODEL
# ============================================================================

@dataclass
class BuildingParams:
    """Parameters for building thermal model"""
    floor_area: float = 1000.0  # m²
    thermal_mass: float = 50000.0  # kg (effective mass: concrete, furnishings)
    specific_heat: float = 1000.0  # J/(kg·K) - effective
    UA_envelope: float = 500.0  # W/K - building heat loss coefficient
    internal_gains: float = 2000.0  # W - people, equipment, lighting
    setpoint_temp: float = 21.0  # °C - desired temperature
    deadband: float = 1.0  # °C - thermostat deadband
    solar_aperture: float = 50.0  # m² - effective window area
    solar_gain_factor: float = 0.6  # SHGC - solar heat gain coefficient


class BuildingThermalMass(FluidComponent):
    """
    Building with thermal mass, heat losses, and internal gains.
    More realistic than simple load profile.
    """

    def __init__(self, name: str, params: BuildingParams, initial_temp: float = 20.0):
        super().__init__(name)
        self.params = params
        self.T_building = initial_temp

        # Heating demand state
        self.Q_heating_required = 0.0
        self.heating_active = False

    def calculate_heat_losses(self, T_ambient: float) -> float:
        """Calculate building heat loss to environment"""
        Q_loss = self.params.UA_envelope * (self.T_building - T_ambient)
        return Q_loss

    def calculate_solar_gains(self, irradiance: float) -> float:
        """Calculate solar heat gain through windows"""
        Q_solar = irradiance * self.params.solar_aperture * self.params.solar_gain_factor
        return Q_solar

    def calculate_heating_demand(self) -> float:
        """Determine if heating is needed based on setpoint control"""
        T_error = self.params.setpoint_temp - self.T_building

        # Thermostat with deadband
        if not self.heating_active:
            # Turn on if below setpoint - deadband
            if T_error > self.params.deadband:
                self.heating_active = True
        else:
            # Turn off if above setpoint
            if T_error < 0:
                self.heating_active = False

        # Calculate required heating power (proportional to error when active)
        if self.heating_active:
            # Estimate required power to reach setpoint
            # Simple proportional control
            K_p = 2000.0  # W/K - proportional gain
            Q_required = K_p * T_error
            return max(Q_required, 0.0)
        else:
            return 0.0

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update building thermal state.

        Inputs:
            - T_inlet: Supply temperature from heating system (°C)
            - flow_rate: Heating water flow rate (kg/s)
            - T_ambient: Outside air temperature (°C)
            - irradiance: Solar irradiance (W/m²)
            - fluid_cp: Fluid specific heat (J/(kg·K))
        """
        T_inlet = inputs.get('T_inlet', self.T_building)
        flow_rate = inputs.get('flow_rate', 0.0)
        T_ambient = inputs.get('T_ambient', 20.0)
        irradiance = inputs.get('irradiance', 0.0)
        fluid_cp = inputs.get('fluid_cp', 4186.0)

        # Calculate heating demand
        self.Q_heating_required = self.calculate_heating_demand()

        # Heat delivered by heating system
        # Assume heat exchanger in building: fluid cools down as it heats the space
        T_outlet = self.calculate_outlet_temp(T_inlet, flow_rate, dt)
        Q_delivered = flow_rate * fluid_cp * (T_inlet - T_outlet) if flow_rate > 0 else 0

        # Building heat balance
        Q_loss = self.calculate_heat_losses(T_ambient)
        Q_solar = self.calculate_solar_gains(irradiance)
        Q_internal = self.params.internal_gains

        # Net heat to building
        Q_net = Q_delivered + Q_solar + Q_internal - Q_loss

        # Update building temperature
        dT = (Q_net * dt) / (self.params.thermal_mass * self.params.specific_heat)
        self.T_building += dT

        return {
            'T_outlet': T_outlet,
            'Q_demand': self.Q_heating_required,
            'Q_actual': Q_delivered,
            'Q_loss': Q_loss,
            'Q_solar_gain': Q_solar,
            'T_building': self.T_building,
            'heating_active': self.heating_active,
            'load_fraction': Q_delivered / self.Q_heating_required if self.Q_heating_required > 0 else 1.0
        }

    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """
        Calculate heating system return temperature.
        Based on required heat extraction and available flow.
        """
        if flow_rate < 1e-6:
            return T_inlet

        # Heat that can be extracted
        fluid_cp = 4186.0
        max_dT = 20.0  # Maximum temperature drop

        # Calculate required temperature drop to meet demand
        dT_required = self.Q_heating_required / (flow_rate * fluid_cp)
        dT = min(dT_required, max_dT)

        # Also limited by building temperature (can't go below building temp)
        T_outlet = max(T_inlet - dT, self.T_building)

        return T_outlet

    def get_state(self) -> Dict[str, Any]:
        return {
            'T_building': self.T_building,
            'Q_demand': self.Q_heating_required,
            'heating_active': self.heating_active,
            'floor_area': self.params.floor_area
        }
