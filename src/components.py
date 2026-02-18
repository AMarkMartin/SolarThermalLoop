"""
Modular component library for solar-assisted heat pump (SAHP) system.
Each component can be independently upgraded while maintaining system integration.

System architecture:
  Solar loop: [Storage Tank] → [Pump] → [Unglazed Collector] → [Storage Tank]
  Load side:  [Storage Tank] → [Heat Pump evaporator] → [Building HVAC]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


# ============================================================================
# FLUID PROPERTIES
# ============================================================================

@dataclass
class WaterGlycolMixture:
    """
    Thermophysical properties for propylene glycol / water mixture.
    Default is 30% propylene glycol by volume — provides freeze protection
    to approximately -15°C while maintaining reasonable heat transfer.
    """
    glycol_fraction: float = 0.30     # volume fraction propylene glycol
    density: float = 1030.0           # kg/m³ at ~20°C
    specific_heat: float = 3700.0     # J/(kg·K) at ~20°C
    freeze_point: float = -15.0       # °C

    def cp_at_temp(self, T: float) -> float:
        """
        Specific heat with mild temperature dependence.
        Glycol mixtures have slightly increasing cp with temperature.
        Linear approximation valid for -10°C to 80°C range.
        """
        # ~+0.8 J/(kg·K) per °C above 20°C reference
        return self.specific_heat + 0.8 * (T - 20.0)


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
    """
    Parameters for unglazed solar thermal collector.

    Unglazed collectors have no glass cover, resulting in higher heat loss
    coefficients (10-25 W/(m²·K)) but simpler construction and the ability
    to absorb heat from ambient air when collector temp < ambient.
    Well-suited as a source booster for heat pump systems where high
    fluid temperatures are not required.
    """
    area: float = 300.0               # m² — commercial roof-mounted array
    efficiency: float = 0.90          # Optical efficiency (no glazing transmission loss)
    absorptance: float = 0.92         # Absorber surface absorptance (dark polymer/metal)
    heat_loss_coef: float = 15.0      # W/(m²·K) — unglazed, wind-exposed
    thermal_mass: float = 3000.0      # kg (absorber material + fluid in array)
    specific_heat: float = 800.0      # J/(kg·K) — effective (absorber + glycol mix)


class SolarCollector(FluidComponent):
    """
    Unglazed solar thermal collector with physics-based heat transfer.
    Higher heat losses than glazed collectors, but can harvest ambient
    heat when collector temperature is below ambient (net heat gain
    from convection). Suitable for SAHP source-side boosting.
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

        # Heat transferred to fluid (computed before thermal mass update)
        T_outlet = self.calculate_outlet_temp(T_inlet, flow_rate, dt)
        Q_to_fluid = flow_rate * fluid_cp * (T_outlet - T_inlet) if flow_rate > 0 else 0.0

        # Net heat to collector thermal mass:
        # absorber gains solar, loses to ambient AND to the fluid
        Q_net = Q_absorbed - Q_loss - Q_to_fluid

        # Update collector temperature (thermal mass effect)
        dT_collector = (Q_net * dt) / (self.params.thermal_mass * self.params.specific_heat)
        self.T_collector += dT_collector

        return {
            'T_outlet': T_outlet,
            'Q_collected': Q_absorbed - Q_loss,
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
    """
    Parameters for insulated thermal energy storage tank.
    Stores water-glycol mixture for the solar loop / heat pump source side.
    Supports vertical stratification via num_nodes (1 = fully mixed).
    """
    volume: float = 15.0              # m³ (15,000 L)
    mass: float = 15450.0             # kg (glycol mixture at ~1030 kg/m³)
    specific_heat: float = 3700.0     # J/(kg·K) — water-glycol mixture
    surface_area: float = 30.0        # m² (cylindrical ~2.5m dia × 3m tall)
    heat_loss_coef: float = 0.5       # W/(m²·K) — well-insulated (~100mm foam)
    num_nodes: int = 1                # 1 = fully mixed, >=2 = stratified
    tank_height: float = 3.0          # m (vertical dimension of cylinder)
    k_eff: float = 0.6               # W/(m·K) effective axial conductivity


class StorageTank(FluidComponent):
    """
    Insulated thermal energy storage tank with optional vertical stratification.

    Node convention: index 0 = TOP of tank, index N-1 = BOTTOM.
    Solar hot return enters the top; solar cold draw exits the bottom.
    Heat pump / load draws warm fluid from top; returns cool fluid to bottom.

    When num_nodes=1, behavior is identical to the original fully-mixed model.
    """

    def __init__(self, name: str, params: StorageTankParams, initial_temp: float = 20.0):
        super().__init__(name)
        self.params = params
        self.num_nodes = params.num_nodes

        # Temperature state: numpy array of node temperatures (top to bottom)
        self.T_nodes = np.full(self.num_nodes, initial_temp, dtype=float)

        # Derived geometry
        self._node_mass = params.mass / self.num_nodes
        self._node_height = params.tank_height / self.num_nodes
        self._A_cross = params.volume / params.tank_height
        self._node_surface_area = params.surface_area / self.num_nodes

    @property
    def T_tank(self) -> float:
        """Backward-compatible scalar temperature (arithmetic mean of all nodes)."""
        return float(np.mean(self.T_nodes))

    @T_tank.setter
    def T_tank(self, value: float):
        """Allow direct assignment for backward compatibility (sets all nodes)."""
        self.T_nodes[:] = value

    @property
    def T_tank_top(self) -> float:
        """Temperature of the topmost node (warmest in stratified state)."""
        return float(self.T_nodes[0])

    @property
    def T_tank_bottom(self) -> float:
        """Temperature of the bottommost node (coolest in stratified state)."""
        return float(self.T_nodes[-1])

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
        fluid_cp = inputs.get('fluid_cp', 3700.0)

        N = self.num_nodes
        cp = self.params.specific_heat

        # ---- FAST PATH: single node (original fully-mixed model) ----
        if N == 1:
            T = self.T_nodes[0]
            Q_solar = flow_rate_solar * fluid_cp * (T_inlet_solar - T)
            Q_load = flow_rate_load * fluid_cp * (T_inlet_load - T)
            Q_loss = self.params.heat_loss_coef * self.params.surface_area * (T - T_ambient)
            Q_net = Q_solar + Q_load - Q_loss
            self.T_nodes[0] += (Q_net * dt) / (self.params.mass * cp)

            T_avg = self.T_tank
            return {
                'T_tank': T_avg,
                'T_outlet_solar': T_avg,
                'T_outlet_load': T_avg,
                'Q_solar': Q_solar,
                'Q_load': Q_load,
                'Q_loss': Q_loss,
                'energy_stored': self.params.mass * cp * (T_avg - 20.0) / 1e6,
                'T_tank_top': T_avg,
                'T_tank_bottom': T_avg,
                'T_nodes': self.T_nodes.copy(),
                'stratification_dT': 0.0,
            }

        # ---- MULTI-NODE STRATIFIED MODEL ----
        node_mass = self._node_mass
        dT_nodes = np.zeros(N)

        # Outlet temperatures (read BEFORE state update)
        T_outlet_solar = float(self.T_nodes[-1])   # cold draw from bottom
        T_outlet_load = float(self.T_nodes[0])      # warm draw from top

        # (a) Solar loop advection: hot enters top (node 0), exits bottom (node N-1)
        #     Flow direction: top → bottom through nodes
        if flow_rate_solar > 1e-9:
            for i in range(N):
                T_upstream = T_inlet_solar if i == 0 else self.T_nodes[i - 1]
                Q_advect = flow_rate_solar * fluid_cp * (T_upstream - self.T_nodes[i])
                dT_nodes[i] += (Q_advect * dt) / (node_mass * cp)

        # (b) Load loop advection: cool return enters bottom (node N-1), exits top (node 0)
        #     Flow direction: bottom → top through nodes
        #     Use abs() to handle both positive and negative flow_rate_load conventions
        fr_load = abs(flow_rate_load)
        if fr_load > 1e-9:
            for i in range(N - 1, -1, -1):
                T_upstream = T_inlet_load if i == N - 1 else self.T_nodes[i + 1]
                Q_advect = fr_load * fluid_cp * (T_upstream - self.T_nodes[i])
                dT_nodes[i] += (Q_advect * dt) / (node_mass * cp)

        # (c) Inter-node conduction
        cond_coeff = self.params.k_eff * self._A_cross / self._node_height  # W/K
        for i in range(N - 1):
            Q_cond = cond_coeff * (self.T_nodes[i] - self.T_nodes[i + 1])
            dT_cond = (Q_cond * dt) / (node_mass * cp)
            dT_nodes[i] -= dT_cond
            dT_nodes[i + 1] += dT_cond

        # (d) Heat loss to ambient (per node)
        Q_loss_total = 0.0
        for i in range(N):
            Q_loss_i = self.params.heat_loss_coef * self._node_surface_area * (self.T_nodes[i] - T_ambient)
            Q_loss_total += Q_loss_i
            dT_nodes[i] -= (Q_loss_i * dt) / (node_mass * cp)

        # (e) Apply temperature changes
        self.T_nodes += dT_nodes

        # (f) Buoyancy mixing — enforce stable stratification (hot on top)
        self._apply_buoyancy_mixing()

        # Aggregate output quantities (match original sign conventions)
        Q_solar = flow_rate_solar * fluid_cp * (T_inlet_solar - T_outlet_solar)
        Q_load = flow_rate_load * fluid_cp * (T_inlet_load - T_outlet_load)

        T_avg = self.T_tank

        return {
            'T_tank': T_avg,
            'T_outlet_solar': T_outlet_solar,
            'T_outlet_load': T_outlet_load,
            'Q_solar': Q_solar,
            'Q_load': Q_load,
            'Q_loss': Q_loss_total,
            'energy_stored': self.params.mass * cp * (T_avg - 20.0) / 1e6,
            'T_tank_top': self.T_tank_top,
            'T_tank_bottom': self.T_tank_bottom,
            'T_nodes': self.T_nodes.copy(),
            'stratification_dT': self.T_tank_top - self.T_tank_bottom,
        }

    def _apply_buoyancy_mixing(self):
        """Enforce stable stratification (hot on top, cold on bottom).
        Uses a stack-based algorithm that resolves all inversions in one O(n) pass.
        Groups of nodes with density inversions are mixed to their mass-weighted average."""
        N = self.num_nodes
        if N < 2:
            return

        # Stack entries: (sum_of_temperatures, count_of_nodes)
        stack = []
        for i in range(N):
            stack.append((self.T_nodes[i], 1))
            # Merge with group above if current group is warmer (inversion)
            while len(stack) >= 2:
                top_sum, top_count = stack[-1]
                prev_sum, prev_count = stack[-2]
                if prev_sum / prev_count < top_sum / top_count:
                    stack.pop()
                    stack.pop()
                    stack.append((prev_sum + top_sum, prev_count + top_count))
                else:
                    break

        # Reconstruct T_nodes from stack
        idx = 0
        for group_sum, group_count in stack:
            avg = group_sum / group_count
            for _ in range(group_count):
                self.T_nodes[idx] = avg
                idx += 1

    def calculate_outlet_temp(self, T_inlet: float, flow_rate: float, dt: float) -> float:
        """For generic access, returns average tank temperature."""
        return self.T_tank

    def get_state(self) -> Dict[str, Any]:
        state = {
            'T_tank': self.T_tank,
            'volume': self.params.volume,
            'mass': self.params.mass,
            'num_nodes': self.num_nodes,
        }
        if self.num_nodes > 1:
            state['T_nodes'] = self.T_nodes.copy()
            state['T_tank_top'] = self.T_tank_top
            state['T_tank_bottom'] = self.T_tank_bottom
            state['stratification_dT'] = self.T_tank_top - self.T_tank_bottom
        return state


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
        fluid_cp = inputs.get('fluid_cp', 3700.0)

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
        fluid_cp = 3700.0  # J/(kg·K) - water-glycol mixture
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

    def __init__(self, name: str, max_flow: float = 5.0):
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

    def __init__(self, name: str, max_flow: float = 5.0):
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


# ============================================================================
# PUMP WITH PERFORMANCE CURVES
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
# HEAT PUMP COMPONENT (simplified boundary load on tank)
# ============================================================================

@dataclass
class HeatPumpParams:
    """
    Parameters for simplified heat pump model.
    The heat pump draws heat from the storage tank (evaporator side)
    and delivers it at higher temperature to the building (condenser side).
    Modeled as a boundary load on the solar loop — not the simulation focus.
    """
    heating_capacity: float = 300000.0   # W (300 kW rated heating output)
    carnot_efficiency: float = 0.45      # Fraction of Carnot COP achieved (typical 0.4-0.5)
    T_supply: float = 45.0              # °C — condenser-side supply temperature to building
    min_source_temp: float = -10.0       # °C — minimum evaporator source temperature


class HeatPump(Component):
    """
    Simplified heat pump model acting as a heat sink on the storage tank.
    Uses Carnot-based COP calculation from tank temperature to determine
    how much heat is extracted from the tank for a given heating demand.

    Energy balance: Q_heating = Q_evaporator + W_compressor
    COP = Q_heating / W_compressor
    """

    def __init__(self, name: str, params: HeatPumpParams = None):
        super().__init__(name)
        self.params = params or HeatPumpParams()
        self.COP = 0.0
        self.W_compressor = 0.0
        self.Q_heating = 0.0
        self.Q_evaporator = 0.0

    def calculate_cop(self, T_source: float) -> float:
        """
        Calculate COP using Carnot efficiency fraction.
        COP_carnot = T_hot / (T_hot - T_cold)  [in Kelvin]
        COP_actual = carnot_efficiency * COP_carnot
        """
        T_hot_K = self.params.T_supply + 273.15
        T_cold_K = T_source + 273.15

        dT = T_hot_K - T_cold_K
        if dT < 1.0:
            # Source is near or above supply temp — COP capped at high value
            return 8.0

        COP_carnot = T_hot_K / dT
        COP = self.params.carnot_efficiency * COP_carnot
        return np.clip(COP, 1.5, 8.0)

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update heat pump state.

        Inputs expected:
            - T_tank: Storage tank temperature (°C) — evaporator source
            - Q_demand: Building heating demand (W)
        """
        T_tank = inputs.get('T_tank', 20.0)
        Q_demand = inputs.get('Q_demand', 0.0)

        if T_tank < self.params.min_source_temp or Q_demand <= 0:
            self.COP = 0.0
            self.W_compressor = 0.0
            self.Q_heating = 0.0
            self.Q_evaporator = 0.0
        else:
            self.COP = self.calculate_cop(T_tank)

            # Deliver up to rated capacity
            self.Q_heating = min(Q_demand, self.params.heating_capacity)

            # Compressor work and evaporator heat from energy balance
            self.W_compressor = self.Q_heating / self.COP
            self.Q_evaporator = self.Q_heating - self.W_compressor

        return {
            'Q_heating': self.Q_heating,
            'W_compressor': self.W_compressor,
            'Q_evaporator': self.Q_evaporator,
            'COP': self.COP,
            'Q_extracted_from_tank': self.Q_evaporator,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'COP': self.COP,
            'Q_heating': self.Q_heating,
            'W_compressor': self.W_compressor,
            'Q_evaporator': self.Q_evaporator,
        }
