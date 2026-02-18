"""
Control system for solar thermal system.
Manages pumps and valves based on system state.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class Controller(ABC):
    """Base class for control strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute control actions based on system state.

        Args:
            system_state: Dictionary containing current state of all components

        Returns:
            Dictionary of control commands for actuators (pumps, valves)
        """
        pass


class BasicController(Controller):
    """
    Basic rule-based controller for solar thermal system.
    Can be upgraded to PID, MPC, or other advanced control strategies.
    """

    def __init__(
        self,
        name: str = "BasicController",
        solar_dT_threshold: float = 5.0,  # °C - minimum dT to run solar pump
        tank_max_temp: float = 85.0,  # °C - max tank temperature
        load_supply_temp: float = 45.0,  # °C - target supply temperature to load
    ):
        super().__init__(name)
        self.solar_dT_threshold = solar_dT_threshold
        self.tank_max_temp = tank_max_temp
        self.load_supply_temp = load_supply_temp

        # Control state
        self.solar_pump_on = False
        self.load_pump_on = True  # Load typically always runs

    def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute control actions using differential temperature control.

        System state expected:
            - T_collector: Solar collector temperature
            - T_tank: Storage tank temperature
            - Q_demand: Current thermal load demand
            - irradiance: Solar irradiance
        """
        T_collector = system_state.get('T_collector', 20.0)
        T_tank = system_state.get('T_tank', 20.0)
        Q_demand = system_state.get('Q_demand', 0.0)
        irradiance = system_state.get('irradiance', 0.0)

        # Solar pump control: differential temperature control with hysteresis
        dT = T_collector - T_tank

        if not self.solar_pump_on:
            # Turn on if dT exceeds threshold and tank not at max
            if dT > self.solar_dT_threshold and T_tank < self.tank_max_temp and irradiance > 100:
                self.solar_pump_on = True
        else:
            # Turn off if dT drops below threshold or tank too hot
            if dT < self.solar_dT_threshold * 0.5 or T_tank >= self.tank_max_temp or irradiance < 50:
                self.solar_pump_on = False

        # Solar pump speed (can vary based on dT)
        if self.solar_pump_on:
            # Variable speed based on temperature difference
            solar_pump_speed = np.clip(0.3 + (dT - self.solar_dT_threshold) / 20.0, 0.3, 1.0)
        else:
            solar_pump_speed = 0.0

        # Load pump control: modulate based on demand and available temperature
        if Q_demand > 0 and T_tank > 30.0:
            # Calculate required flow rate to meet load
            # Q = m_dot * cp * dT
            fluid_cp = 4186.0  # J/(kg·K)
            dT_load = 15.0  # Target temperature drop across load
            required_flow = Q_demand / (fluid_cp * dT_load)

            # Limit to reasonable values
            load_pump_speed = np.clip(required_flow / 0.05, 0.1, 1.0)
            self.load_pump_on = True
        else:
            load_pump_speed = 0.1  # Minimum circulation
            self.load_pump_on = False

        # Valve positions (for future use - currently just follow pump state)
        solar_valve_position = 1.0 if self.solar_pump_on else 0.0
        load_valve_position = 1.0 if self.load_pump_on else 0.0

        return {
            'solar_pump_speed': solar_pump_speed,
            'load_pump_speed': load_pump_speed,
            'solar_valve_position': solar_valve_position,
            'load_valve_position': load_valve_position,
            'control_state': {
                'solar_pump_on': self.solar_pump_on,
                'load_pump_on': self.load_pump_on,
                'dT_solar': dT
            }
        }


class PIDController(Controller):
    """
    PID controller template for more advanced control.
    Can be used to control tank temperature or load supply temperature.
    """

    def __init__(
        self,
        name: str,
        Kp: float = 1.0,
        Ki: float = 0.1,
        Kd: float = 0.05,
        setpoint: float = 50.0
    ):
        super().__init__(name)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.integral = 0.0
        self.prev_error = 0.0

    def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        PID control implementation.
        Override this method to implement specific control loops.
        """
        # Get process variable (e.g., tank temperature)
        pv = system_state.get('T_tank', 20.0)
        dt = system_state.get('dt', 60.0)

        # Calculate error
        error = self.setpoint - pv

        # PID terms
        P = self.Kp * error
        self.integral += error * dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / dt if dt > 0 else 0.0

        # Control output
        output = P + I + D

        # Anti-windup: limit integral
        self.integral = np.clip(self.integral, -100, 100)

        # Store for next iteration
        self.prev_error = error

        # Saturate output to valid range [0, 1]
        control_signal = np.clip(output, 0.0, 1.0)

        return {
            'control_output': control_signal,
            'error': error,
            'P': P,
            'I': I,
            'D': D
        }


class OptimizingController(Controller):
    """
    Template for optimization-based controller (e.g., MPC).
    Can be upgraded to implement model predictive control.
    """

    def __init__(self, name: str, prediction_horizon: int = 12):
        super().__init__(name)
        self.prediction_horizon = prediction_horizon

    def compute_control(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimization-based control - placeholder for future implementation.
        Could include:
        - Weather forecast integration
        - Load prediction
        - Cost optimization (e.g., minimize auxiliary heating)
        """
        # Placeholder - use basic control for now
        basic_controller = BasicController()
        return basic_controller.compute_control(system_state)
