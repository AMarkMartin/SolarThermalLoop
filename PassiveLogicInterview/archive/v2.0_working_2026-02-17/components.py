"""
Modular component library for solar thermal system.
Each component can be independently upgraded while maintaining system integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


# ============================================================================
# BASE CLASSES - Define interfaces for all components
# ============================================================================

class Component(ABC):
    """Base class for all system components"""

    def __init__(self, name: str):
        self.name = name
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update component state for one timestep.

        Args:
            dt: Time step in seconds
            inputs: Dictionary of input values from connected components

        Returns:
            Dictionary of output values for other components
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current component state for monitoring/logging"""
        pass

    def reset(self):
        """Reset component to initial state"""
        self._state = {}


class FluidComponent(Component):
    """Base class for components that handle fluid flow"""

    @abstractmethod
    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """Calculate outlet fluid temperature"""
        pass


# ============================================================================
# SOLAR COLLECTOR COMPONENT
# ============================================================================

@dataclass
class SolarCollectorParams:
    """Parameters for solar collector physics model"""
    area: float = 2.0  # m²
    efficiency: float = 0.75  # Optical efficiency
    absorptance: float = 0.95
    heat_loss_coef: float = 5.0  # W/(m²·K)
    thermal_mass: float = 10.0  # kg (collector + fluid)
    specific_heat: float = 500.0  # J/(kg·K) - effective


class SolarCollector(FluidComponent):
    """
    Solar thermal collector with physics-based heat transfer.
    Can be upgraded with more sophisticated models (angle-dependent efficiency, etc.)
    """

    def __init__(self, name: str, params: SolarCollectorParams):
        super().__init__(name)
        self.params = params
        self.T_collector = 20.0  # Average collector temperature (°C)

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update collector state.

        Inputs expected:
            - T_inlet: Inlet fluid temperature (°C)
            - flow_rate: Mass flow rate (kg/s)
            - irradiance: Solar irradiance (W/m²)
            - T_ambient: Ambient temperature (°C)
            - fluid_cp: Fluid specific heat (J/(kg·K))
        """
        T_inlet = inputs.get('T_inlet', 20.0)
        flow_rate = inputs.get('flow_rate', 0.0)
        irradiance = inputs.get('irradiance', 0.0)
        T_ambient = inputs.get('T_ambient', 20.0)
        fluid_cp = inputs.get('fluid_cp', 3500.0)

        # Heat absorbed from solar radiation
        Q_absorbed = (irradiance * self.params.area *
                     self.params.absorptance * self.params.efficiency)

        # Heat loss to ambient
        Q_loss = (self.params.heat_loss_coef * self.params.area *
                 (self.T_collector - T_ambient))

        # Net heat to collector
        Q_net = Q_absorbed - Q_loss

        # Update collector temperature (thermal mass effect)
        dT_collector = (Q_net * dt) / (self.params.thermal_mass * self.params.specific_heat)
        self.T_collector += dT_collector

        # Calculate outlet temperature based on flow
        T_outlet = self.calculate_outlet_temp(T_inlet, flow_rate, dt)

        # Heat transferred to fluid
        Q_to_fluid = flow_rate * fluid_cp * (T_outlet - T_inlet) if flow_rate > 0 else 0

        return {
            'T_outlet': T_outlet,
            'Q_collected': Q_net,
            'Q_to_fluid': Q_to_fluid,
            'T_collector': self.T_collector
        }

    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """Calculate outlet temperature assuming well-mixed collector"""
        if flow_rate < 1e-6:
            return self.T_collector

        # Outlet approaches collector temperature based on residence time
        # Simple mixing model - can be upgraded to more sophisticated heat exchanger model
        alpha = 0.8  # Heat transfer effectiveness
        return T_inlet + alpha * (self.T_collector - T_inlet)

    def get_state(self) -> Dict[str, Any]:
        return {
            'T_collector': self.T_collector,
            'area': self.params.area
        }


# ============================================================================
# STORAGE TANK COMPONENT
# ============================================================================

@dataclass
class StorageTankParams:
    """Parameters for storage tank physics model"""
    volume: float = 0.15  # m³
    mass: float = 150.0  # kg
    specific_heat: float = 4186.0  # J/(kg·K)
    surface_area: float = 1.5  # m²
    heat_loss_coef: float = 1.5  # W/(m²·K)
    num_nodes: int = 1  # For stratification (1 = fully mixed)


class StorageTank(FluidComponent):
    """
    Thermal storage tank with heat capacity and losses.
    Can be upgraded to stratified model by increasing num_nodes.
    """

    def __init__(self, name: str, params: StorageTankParams, initial_temp: float = 20.0):
        super().__init__(name)
        self.params = params

        # Initialize temperature nodes (currently single node = fully mixed)
        self.T_tank = initial_temp

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update tank state.

        Inputs expected:
            - T_inlet_solar: Temperature from solar collector (°C)
            - flow_rate_solar: Flow rate from solar loop (kg/s)
            - T_inlet_load: Temperature from load return (°C)
            - flow_rate_load: Flow rate from load (kg/s)
            - T_ambient: Ambient temperature (°C)
            - fluid_cp: Fluid specific heat (J/(kg·K))
        """
        T_inlet_solar = inputs.get('T_inlet_solar', self.T_tank)
        flow_rate_solar = inputs.get('flow_rate_solar', 0.0)
        T_inlet_load = inputs.get('T_inlet_load', self.T_tank)
        flow_rate_load = inputs.get('flow_rate_load', 0.0)
        T_ambient = inputs.get('T_ambient', 20.0)
        fluid_cp = inputs.get('fluid_cp', 4186.0)

        # Heat delivered by solar loop
        Q_solar = flow_rate_solar * fluid_cp * (T_inlet_solar - self.T_tank)

        # Heat extracted by load loop
        Q_load = flow_rate_load * fluid_cp * (T_inlet_load - self.T_tank)

        # Heat loss to ambient
        Q_loss = self.params.heat_loss_coef * self.params.surface_area * (self.T_tank - T_ambient)

        # Net heat change
        Q_net = Q_solar + Q_load - Q_loss

        # Update tank temperature
        dT = (Q_net * dt) / (self.params.mass * self.params.specific_heat)
        self.T_tank += dT

        return {
            'T_tank': self.T_tank,
            'T_outlet_solar': self.T_tank,  # Return to solar loop
            'T_outlet_load': self.T_tank,   # Supply to load
            'Q_solar': Q_solar,
            'Q_load': Q_load,
            'Q_loss': Q_loss,
            'energy_stored': self.params.mass * self.params.specific_heat * (self.T_tank - 20.0) / 1e6  # MJ
        }

    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """For fully mixed tank, outlet equals tank temperature"""
        return self.T_tank

    def get_state(self) -> Dict[str, Any]:
        return {
            'T_tank': self.T_tank,
            'volume': self.params.volume,
            'mass': self.params.mass
        }


# ============================================================================
# THERMAL LOAD COMPONENT
# ============================================================================

@dataclass
class ThermalLoadParams:
    """Parameters for thermal load model"""
    base_load: float = 500.0  # W - constant base load
    peak_load: float = 2000.0  # W - peak load
    load_profile: str = 'constant'  # 'constant', 'variable', 'scheduled'


class ThermalLoad(FluidComponent):
    """
    Thermal load/demand component.
    Can be upgraded with more sophisticated load profiles.
    """

    def __init__(self, name: str, params: ThermalLoadParams):
        super().__init__(name)
        self.params = params
        self.Q_demand = params.base_load

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update load demand.

        Inputs expected:
            - T_inlet: Supply temperature from tank (°C)
            - flow_rate: Flow rate through load (kg/s)
            - time_hours: Current time in hours (for scheduled loads)
            - fluid_cp: Fluid specific heat (J/(kg·K))
        """
        T_inlet = inputs.get('T_inlet', 40.0)
        flow_rate = inputs.get('flow_rate', 0.0)
        time_hours = inputs.get('time_hours', 0.0)
        fluid_cp = inputs.get('fluid_cp', 4186.0)

        # Calculate load based on profile
        self.Q_demand = self._calculate_load(time_hours)

        # Calculate outlet temperature based on heat extraction
        T_outlet = self.calculate_outlet_temp(T_inlet, flow_rate, dt)

        # Actual heat extracted (limited by available temperature and flow)
        Q_actual = flow_rate * fluid_cp * (T_inlet - T_outlet) if flow_rate > 0 else 0

        return {
            'T_outlet': T_outlet,
            'Q_demand': self.Q_demand,
            'Q_actual': Q_actual,
            'load_fraction': Q_actual / self.Q_demand if self.Q_demand > 0 else 1.0
        }

    def _calculate_load(self, time_hours: float) -> float:
        """Calculate load based on time and profile type"""
        if self.params.load_profile == 'constant':
            return self.params.base_load

        elif self.params.load_profile == 'variable':
            # Variable load: higher during evening/morning
            hour = time_hours % 24
            if 6 <= hour <= 9 or 17 <= hour <= 22:
                return self.params.peak_load
            else:
                return self.params.base_load

        elif self.params.load_profile == 'scheduled':
            # Custom scheduled load - can be enhanced
            hour = time_hours % 24
            # Morning peak
            if 6 <= hour <= 8:
                return self.params.peak_load
            # Evening peak
            elif 18 <= hour <= 21:
                return self.params.peak_load * 1.5
            else:
                return self.params.base_load * 0.5

        return self.params.base_load

    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """Calculate outlet temperature based on heat extraction"""
        if flow_rate < 1e-6:
            return T_inlet

        # Assume load requires specific temperature drop
        # Q = m_dot * cp * dT  =>  dT = Q / (m_dot * cp)
        fluid_cp = 4186.0  # J/(kg·K) - water
        dT = self.Q_demand / (flow_rate * fluid_cp)

        # Limit temperature drop to realistic values
        dT = min(dT, 20.0)  # Max 20°C drop

        return T_inlet - dT

    def get_state(self) -> Dict[str, Any]:
        return {
            'Q_demand': self.Q_demand,
            'profile': self.params.load_profile
        }


# ============================================================================
# VALVE COMPONENT
# ============================================================================

class Valve(Component):
    """
    Flow control valve with adjustable position.
    Can be upgraded with more sophisticated valve characteristics.
    """

    def __init__(self, name: str, max_flow: float = 0.1):
        super().__init__(name)
        self.max_flow = max_flow  # kg/s
        self.position = 1.0  # 0-1 (fully closed to fully open)
        self.flow_rate = 0.0

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update valve flow.

        Inputs expected:
            - position: Commanded valve position (0-1)
            - pressure_diff: Pressure difference across valve (Pa) [optional]
        """
        self.position = np.clip(inputs.get('position', self.position), 0.0, 1.0)

        # Simple linear valve characteristic (can be upgraded to non-linear)
        self.flow_rate = self.position * self.max_flow

        return {
            'flow_rate': self.flow_rate,
            'position': self.position
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'flow_rate': self.flow_rate
        }


# ============================================================================
# PUMP COMPONENT
# ============================================================================

class Pump(Component):
    """
    Circulation pump with variable speed.
    Can be upgraded with pump curves, efficiency, power consumption.
    """

    def __init__(self, name: str, max_flow: float = 0.1):
        super().__init__(name)
        self.max_flow = max_flow  # kg/s
        self.speed = 1.0  # 0-1 (0% to 100%)
        self.flow_rate = 0.0

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update pump operation.

        Inputs expected:
            - speed: Commanded pump speed (0-1)
        """
        self.speed = np.clip(inputs.get('speed', self.speed), 0.0, 1.0)

        # Simple linear flow-speed relationship (can be upgraded to pump curve)
        self.flow_rate = self.speed * self.max_flow

        return {
            'flow_rate': self.flow_rate,
            'speed': self.speed
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'speed': self.speed,
            'flow_rate': self.flow_rate
        }
