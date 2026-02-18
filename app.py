"""
Solar Thermal System — Interactive Streamlit Demo

Run with:
    streamlit run app.py
"""

import os, sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    Component, PumpWithCurve,
)
from src.models import SolarRadiationModel, LocationParams, WeatherForecast
from src.control import BasicController


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Thermal System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    tank        = "#e74c3c",
    collector   = "#f39c12",
    outdoor     = "#3498db",
    supply      = "#9b59b6",
    return_t    = "#1abc9c",
    solar_del   = "#2ecc71",
    auxiliary   = "#c0392b",
    demand      = "#2c3e50",
    irr_fill    = "rgba(255,200,0,0.12)",
    solar_pump  = "#27ae60",
    load_pump   = "#2980b9",
    direct      = "#e67e22",
    diffuse     = "#95a5a6",
    night_bg    = "rgba(150,160,200,0.07)",
)

TEMPLATE = "plotly_white"


# ─────────────────────────────────────────────────────────────────────────────
# INLINED BUILDING MODEL  (from examples/winter_scenario.py)
# ─────────────────────────────────────────────────────────────────────────────
class SimplifiedBuildingLoad(Component):
    """UA-based building heating load with night setback."""

    def __init__(self, name, floor_area=200.0, design_heat_load=5000.0,
                 design_outdoor_temp=-10.0, indoor_setpoint=21.0):
        super().__init__(name)
        self.floor_area         = floor_area
        self.design_heat_load   = design_heat_load
        self.indoor_setpoint    = indoor_setpoint
        self.UA = design_heat_load / (indoor_setpoint - design_outdoor_temp)
        self.Q_total_demand     = 0.0
        self.Q_solar_delivered  = 0.0
        self.Q_auxiliary        = 0.0
        self.T_return           = indoor_setpoint - 10.0

    def _demand(self, T_outdoor: float, hour: int) -> float:
        Q = self.UA * (self.indoor_setpoint - T_outdoor)
        f = 1.0 if 6 <= int(hour) % 24 <= 22 else 0.7
        return max(Q * f, 0.0)

    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        T_supply  = inputs.get("T_supply",  30.0)
        flow_rate = inputs.get("flow_rate",  0.0)
        T_outdoor = inputs.get("T_outdoor",  0.0)
        hour      = inputs.get("hour",      12)
        fluid_cp  = inputs.get("fluid_cp", 4186.0)

        self.Q_total_demand = self._demand(T_outdoor, hour)

        if flow_rate > 0 and T_supply > self.T_return:
            Q_avail = flow_rate * fluid_cp * (T_supply - self.T_return)
            self.Q_solar_delivered = min(Q_avail, self.Q_total_demand)
            self.T_return = T_supply - self.Q_solar_delivered / (flow_rate * fluid_cp)
        else:
            self.Q_solar_delivered = 0.0
            self.T_return = self.indoor_setpoint - 10.0

        self.Q_auxiliary = max(self.Q_total_demand - self.Q_solar_delivered, 0.0)
        sf = (self.Q_solar_delivered / self.Q_total_demand
              if self.Q_total_demand > 0 else 0.0)
        return {
            "Q_total_demand":    self.Q_total_demand,
            "Q_solar_delivered": self.Q_solar_delivered,
            "Q_auxiliary":       self.Q_auxiliary,
            "solar_fraction":    sf,
            "T_return":          self.T_return,
        }

    def get_state(self) -> Dict[str, Any]:
        return {"Q_demand": self.Q_total_demand, "UA": self.UA}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def night_bands(time_arr: np.ndarray, irr_arr: np.ndarray, threshold: float = 10.0):
    """Return list of (t_start, t_end) tuples covering nighttime periods."""
    is_night = irr_arr < threshold
    bands, in_night, t_start = [], False, 0.0
    for t, n in zip(time_arr, is_night):
        if n and not in_night:
            t_start, in_night = float(t), True
        elif not n and in_night:
            bands.append((t_start, float(t)))
            in_night = False
    if in_night:
        bands.append((t_start, float(time_arr[-1])))
    return bands


def shade_nights(fig, time_arr, irr_arr, rows=None, cols=None):
    """Add translucent night-time bands to a plotly figure."""
    bands = night_bands(time_arr, irr_arr)
    rows  = rows  or [None]
    cols  = cols  or [None]
    for t0, t1 in bands:
        for r, c in zip(rows, cols):
            kw = dict(row=r, col=c) if r else {}
            fig.add_vrect(
                x0=t0, x1=t1,
                fillcolor=C["night_bg"],
                line_width=0,
                **kw,
            )


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulation…")
def run_simulation(cfg_items: tuple) -> Dict[str, np.ndarray]:
    """
    Run a full solar-thermal system simulation.
    Arguments passed as a sorted tuple of (key, value) pairs so that
    st.cache_data can hash them reliably.
    """
    cfg = dict(cfg_items)

    location = LocationParams(
        latitude =cfg["latitude"],
        longitude=cfg["longitude"],
        timezone =cfg["timezone"],
        elevation=cfg["elevation"],
    )

    collector = SolarCollector(
        "SolarCollector",
        SolarCollectorParams(
            area          =cfg["collector_area"],
            efficiency    =cfg["collector_efficiency"],
            heat_loss_coef=cfg["collector_heat_loss"],
        ),
    )

    vol      = cfg["tank_volume"]
    h_tank   = 3.0 * (vol / 15.0) ** (1 / 3)
    sa_tank  = max(6.0 * vol ** (2 / 3), 0.5)

    tank = StorageTank(
        "StorageTank",
        StorageTankParams(
            volume        =vol,
            mass          =vol * 1030,
            surface_area  =sa_tank,
            heat_loss_coef=cfg["tank_loss_coef"],
            num_nodes     =cfg["tank_nodes"],
            tank_height   =h_tank,
        ),
        initial_temp=cfg["tank_init_temp"],
    )

    building = SimplifiedBuildingLoad(
        "Building",
        floor_area         =cfg["floor_area"],
        design_heat_load   =cfg["design_heat_load"],
        design_outdoor_temp=cfg["design_outdoor_temp"],
        indoor_setpoint    =cfg["indoor_setpoint"],
    )

    controller = BasicController(
        solar_dT_threshold=cfg["dT_threshold"],
        tank_max_temp     =cfg["tank_max_temp"],
    )

    solar_pump = PumpWithCurve("SolarPump",
                               rated_flow=0.03, rated_head=4.0,
                               max_head=6.0, efficiency_bep=0.68)
    load_pump  = PumpWithCurve("LoadPump",
                               rated_flow=0.08, rated_head=5.0,
                               max_head=7.5, efficiency_bep=0.72)

    solar_model = SolarRadiationModel(location)
    weather     = WeatherForecast(days=cfg["duration_days"] + 2)

    dt             = cfg["dt"]
    duration_hours = cfg["duration_days"] * 24.0
    start_day      = cfg["start_day"]
    base_temp      = cfg["base_temp"]
    fluid_cp_s     = 3500.0
    fluid_cp_l     = 4186.0
    gas_price      = 0.04
    num_steps      = int(duration_hours * 3600 / dt)

    buf: Dict[str, list] = {k: [] for k in [
        "time", "T_tank", "T_collector", "T_outdoor", "T_supply", "T_return",
        "irradiance_total", "irradiance_direct", "irradiance_diffuse",
        "solar_altitude", "cloud_cover",
        "Q_solar_collected", "Q_building_demand", "Q_solar_delivered", "Q_auxiliary",
        "solar_fraction",
        "solar_pump_speed", "solar_pump_power", "solar_pump_efficiency",
        "load_pump_speed",  "load_pump_power",  "load_pump_efficiency",
        "aux_cost",
    ]}

    def outdoor_temp(t_h: float) -> float:
        hr   = int(t_h)
        day  = int(t_h / 24)
        Tf   = weather.get_temperature(hr)
        Td   = base_temp - 6 * np.cos(2 * np.pi * ((t_h % 24) - 15) / 24)
        Tv   = 3 * np.sin(2 * np.pi * day / 7)
        return 0.5 * Tf + 0.3 * Td + 0.2 * Tv

    for step in range(num_steps):
        t_h   = step * dt / 3600
        doy   = start_day + int(t_h / 24)
        hr    = int(t_h)
        T_out = outdoor_temp(t_h)
        cloud = weather.get_cloud_cover(hr)

        sol = solar_model.calculate_irradiance(
            time_hours   =t_h % 24,
            day_of_year  =doy,
            cloud_cover  =cloud,
            panel_tilt   =cfg["panel_tilt"],
            panel_azimuth=180.0,
        )
        irr = sol["total"]

        ctrl = controller.compute_control({
            "T_collector": collector.T_collector,
            "T_tank":      tank.T_tank,
            "Q_demand":    building.Q_total_demand,
            "irradiance":  irr,
        })

        sp_out = solar_pump.update(dt, {"speed": ctrl["solar_pump_speed"]})
        lp_out = load_pump.update(dt,  {"speed": ctrl["load_pump_speed"]})

        col_out = collector.update(dt, {
            "T_inlet":    tank.T_tank,
            "flow_rate":  sp_out["flow_rate"],
            "irradiance": irr,
            "T_ambient":  T_out,
            "fluid_cp":   fluid_cp_s,
        })
        bld_out = building.update(dt, {
            "T_supply":  tank.T_tank,
            "flow_rate": lp_out["flow_rate"],
            "T_outdoor": T_out,
            "hour":      hr,
            "fluid_cp":  fluid_cp_l,
        })
        tnk_out = tank.update(dt, {
            "T_inlet_solar":   col_out["T_outlet"],
            "flow_rate_solar": sp_out["flow_rate"],
            "T_inlet_load":    bld_out["T_return"],
            "flow_rate_load":  lp_out["flow_rate"],
            "T_ambient":       T_out,
            "fluid_cp":        fluid_cp_s,
        })

        b = buf
        b["time"].append(t_h)
        b["T_tank"].append(tnk_out["T_tank"])
        b["T_collector"].append(col_out["T_collector"])
        b["T_outdoor"].append(T_out)
        b["T_supply"].append(tank.T_tank)
        b["T_return"].append(bld_out["T_return"])
        b["irradiance_total"].append(sol["total"])
        b["irradiance_direct"].append(sol["direct"])
        b["irradiance_diffuse"].append(sol["diffuse"])
        b["solar_altitude"].append(sol["altitude"])
        b["cloud_cover"].append(cloud)
        b["Q_solar_collected"].append(col_out["Q_to_fluid"])
        b["Q_building_demand"].append(bld_out["Q_total_demand"])
        b["Q_solar_delivered"].append(bld_out["Q_solar_delivered"])
        b["Q_auxiliary"].append(bld_out["Q_auxiliary"])
        b["solar_fraction"].append(bld_out["solar_fraction"])
        b["solar_pump_speed"].append(sp_out["speed"])
        b["solar_pump_power"].append(sp_out["power"])
        b["solar_pump_efficiency"].append(sp_out["efficiency"])
        b["load_pump_speed"].append(lp_out["speed"])
        b["load_pump_power"].append(lp_out["power"])
        b["load_pump_efficiency"].append(lp_out["efficiency"])
        b["aux_cost"].append((bld_out["Q_auxiliary"] * dt / 3.6e6) * gas_price)

    return {k: np.array(v) for k, v in buf.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    with st.expander("📍 Location", expanded=False):
        latitude  = st.slider("Latitude (°N)",    25.0,  65.0,  40.0, 0.5)
        longitude = st.slider("Longitude (°E)", -130.0, -65.0, -105.0, 0.5)
        elevation = st.slider("Elevation (m)",      0,  3000,  1600,  50)
        start_day = st.slider("Start Day of Year",  1,   365,    75,   1,
                              help="75 ≈ early March  |  355 ≈ late December")
        base_temp = st.slider("Baseline Outdoor Temp (°C)", -20.0, 15.0, 5.0, 0.5)

    with st.expander("☀️ Solar Collector", expanded=True):
        collector_area       = st.slider("Array Area (m²)",              5.0, 50.0, 15.0, 0.5)
        collector_efficiency = st.slider("Optical Efficiency",           0.50, 0.95, 0.75, 0.01)
        collector_heat_loss  = st.slider("Heat-Loss Coef (W/m²·K)",      1.0, 25.0,  4.0, 0.5)
        panel_tilt           = st.slider("Panel Tilt (°)",                 0,   90,   50,   1)

    with st.expander("🌡️ Storage Tank", expanded=True):
        tank_volume_L  = st.slider("Volume (L)",               200, 5000, 800, 100)
        tank_loss_coef = st.slider("Heat-Loss Coef (W/m²·K)",  0.1,  3.0, 1.0, 0.1)
        tank_init_temp = st.slider("Initial Temperature (°C)", 10.0, 60.0, 35.0, 1.0)
        tank_stratified = st.checkbox("Stratified tank (10 nodes)",
                                      help="Enables vertical temperature stratification")

    with st.expander("🏢 Building", expanded=False):
        floor_area           = st.slider("Floor Area (m²)",          50, 1000, 200,  25)
        design_heat_load_kW  = st.slider("Design Heat Load (kW)",    1.0, 50.0,  5.0,  0.5)
        design_outdoor_temp  = st.slider("Design Outdoor Temp (°C)", -20.0, 0.0, -10.0, 1.0)
        indoor_setpoint      = st.slider("Indoor Setpoint (°C)",     18.0, 25.0,  21.0, 0.5)

    with st.expander("🔧 Controller", expanded=False):
        dT_threshold  = st.slider("Solar ΔT Threshold (°C)", 2.0, 15.0, 5.0, 0.5,
                                  help="Collector must exceed tank by this ΔT to activate solar pump")
        tank_max_temp = st.slider("Max Tank Temp (°C)",      50.0, 95.0, 75.0, 1.0)

    with st.expander("⏱️ Simulation", expanded=False):
        duration_days = st.slider("Duration (days)", 1, 10, 5)
        dt_s = st.select_slider("Time Step", options=[30, 60, 120], value=60,
                                format_func=lambda x: f"{x} s")

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DICT
# ─────────────────────────────────────────────────────────────────────────────
cfg = {
    "latitude":            latitude,
    "longitude":           longitude,
    "timezone":            round(longitude / 15.0),
    "elevation":           float(elevation),
    "start_day":           start_day,
    "base_temp":           base_temp,
    "collector_area":      collector_area,
    "collector_efficiency":collector_efficiency,
    "collector_heat_loss": collector_heat_loss,
    "panel_tilt":          float(panel_tilt),
    "tank_volume":         tank_volume_L / 1000.0,
    "tank_loss_coef":      tank_loss_coef,
    "tank_init_temp":      tank_init_temp,
    "tank_nodes":          10 if tank_stratified else 1,
    "floor_area":          float(floor_area),
    "design_heat_load":    design_heat_load_kW * 1000.0,
    "design_outdoor_temp": design_outdoor_temp,
    "indoor_setpoint":     indoor_setpoint,
    "dT_threshold":        dT_threshold,
    "tank_max_temp":       tank_max_temp,
    "duration_days":       duration_days,
    "dt":                  float(dt_s),
}

if run_btn:
    st.session_state["history"] = run_simulation(tuple(sorted(cfg.items())))
    st.session_state["cfg"]     = cfg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("☀️ Solar Thermal System — Interactive Demo")

if "history" not in st.session_state:
    # ── Welcome screen ────────────────────────────────────────────────────────
    st.info("Configure the system in the sidebar and click **▶ Run Simulation** to begin.")
    st.markdown("""
### Two-loop solar-assisted heating system

```
Solar Irradiance + Ambient Temp
         │
    ┌────▼──────────────┐
    │  Solar Collector  │◄──── cold draw (bottom of tank)
    └────────┬──────────┘
             │ hot fluid
       ┌─────▼──────┐
       │ Solar Pump │   ← solar loop
       └─────┬──────┘
    ┌────────▼──────────┐
    │   Storage Tank    │
    └────────┬──────────┘
             │ warm supply
       ┌─────▼──────┐
       │  Load Pump │   ← load loop
       └─────┬──────┘
    ┌────────▼──────────┐
    │  Building / Load  │
    └────────┬──────────┘
             │ cool return
             └────────────► back to tank (bottom)
```

**Six tabs are available after running the simulation:**

| Tab | Contents |
|-----|----------|
| 🌡️ Temperatures | Tank, collector, outdoor & supply temperatures vs time |
| ⚡ Energy Flows | Heating demand breakdown, cumulative totals, solar fraction |
| ☀️ Solar Radiation | Irradiance components, sun altitude, cloud cover |
| ⚙️ Pumps | Solar & load pump speed, power, and efficiency |
| 🔍 Solar Explorer | Interactive 24-hour irradiance profile for any day/location |
| 🔄 Pump Explorer | Interactive H–Q and efficiency curves |
""")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
h  = st.session_state["history"]
t  = h["time"]
ts = t * 3600  # seconds

E_dem  = float(np.trapz(h["Q_building_demand"], ts)) / 3.6e6
E_sol  = float(np.trapz(h["Q_solar_delivered"], ts)) / 3.6e6
E_aux  = float(np.trapz(h["Q_auxiliary"],       ts)) / 3.6e6
avg_sf = float(np.mean(h["solar_fraction"])) * 100
cost   = float(np.sum(h["aux_cost"]))
Tlo    = float(np.min(h["T_tank"]))
Thi    = float(np.max(h["T_tank"]))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Building Demand",   f"{E_dem:.1f} kWh")
m2.metric("Solar Delivered",   f"{E_sol:.1f} kWh",
          delta=f"{E_sol / E_dem * 100:.0f}% of demand" if E_dem > 0 else None)
m3.metric("Avg Solar Fraction", f"{avg_sf:.1f}%")
m4.metric("Aux Fuel Cost",     f"${cost:.2f}",
          help="Natural gas @ $0.04 / kWh")
m5.metric("Tank Temperature",  f"{Tlo:.0f} – {Thi:.0f} °C")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_temps, tab_energy, tab_solar,
 tab_pumps, tab_sol_exp, tab_pump_exp) = st.tabs([
    "🌡️ Temperatures",
    "⚡ Energy Flows",
    "☀️ Solar Radiation",
    "⚙️ Pumps",
    "🔍 Solar Explorer",
    "🔄 Pump Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEMPERATURES
# ══════════════════════════════════════════════════════════════════════════════
with tab_temps:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
    )

    traces = [
        ("Tank",      "T_tank",      C["tank"],      2.5, "solid"),
        ("Collector", "T_collector", C["collector"], 2.0, "solid"),
        ("Outdoor",   "T_outdoor",   C["outdoor"],   1.5, "solid"),
        ("Supply",    "T_supply",    C["supply"],    1.5, "dot"),
        ("Return",    "T_return",    C["return_t"],  1.5, "dot"),
    ]
    for label, key, clr, w, dash in traces:
        fig.add_trace(go.Scatter(
            x=t, y=h[key], name=label,
            line=dict(color=clr, width=w, dash=dash),
            hovertemplate=f"{label}: %{{y:.1f}} °C<extra></extra>",
        ), row=1, col=1)

    # Irradiance fill in lower panel
    fig.add_trace(go.Scatter(
        x=t, y=h["irradiance_total"], name="Irradiance",
        fill="tozeroy", fillcolor="rgba(255,190,0,0.18)",
        line=dict(color="#f39c12", width=1.2),
        hovertemplate="Irradiance: %{y:.0f} W/m²<extra></extra>",
    ), row=2, col=1)

    shade_nights(fig, t, h["irradiance_total"], rows=[1, 2], cols=[1, 1])

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Irradiance (W/m²)", row=2, col=1)
    fig.update_xaxes(title_text="Simulation Time (hours)", row=2, col=1)
    fig.update_layout(
        template=TEMPLATE, height=560,
        title="System Temperature Evolution",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENERGY FLOWS
# ══════════════════════════════════════════════════════════════════════════════
with tab_energy:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        # --- Instantaneous power breakdown ---
        Q_sol_kW = h["Q_solar_delivered"] / 1000
        Q_aux_kW = h["Q_auxiliary"] / 1000
        Q_dem_kW = h["Q_building_demand"] / 1000

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=t, y=Q_dem_kW, name="Total Demand",
            line=dict(color=C["demand"], width=2, dash="dash"),
            hovertemplate="Demand: %{y:.2f} kW<extra></extra>",
        ))
        # Auxiliary fills from solar up to demand
        fig2.add_trace(go.Scatter(
            x=t, y=Q_sol_kW + Q_aux_kW,
            name="Auxiliary",
            fill="tonexty",
            fillcolor="rgba(192,57,43,0.25)",
            line=dict(color=C["auxiliary"], width=0.5),
            hovertemplate="Auxiliary: %{customdata:.2f} kW<extra></extra>",
            customdata=Q_aux_kW,
            showlegend=True,
        ))
        fig2.add_trace(go.Scatter(
            x=t, y=Q_sol_kW,
            name="Solar Delivered",
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.35)",
            line=dict(color=C["solar_del"], width=1.5),
            hovertemplate="Solar: %{y:.2f} kW<extra></extra>",
        ))
        shade_nights(fig2, t, h["irradiance_total"])
        fig2.update_layout(
            template=TEMPLATE, height=320,
            title="Instantaneous Heating Breakdown",
            xaxis_title="Time (hours)",
            yaxis_title="Power (kW)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Cumulative energy ---
        dt_h   = float(t[1] - t[0]) if len(t) > 1 else 1 / 60
        dt_sec = dt_h * 3600
        cum_dem = np.cumsum(h["Q_building_demand"]) * dt_sec / 3.6e6
        cum_sol = np.cumsum(h["Q_solar_delivered"]) * dt_sec / 3.6e6
        cum_aux = np.cumsum(h["Q_auxiliary"])       * dt_sec / 3.6e6

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t, y=cum_dem, name="Total Demand",
                                  line=dict(color=C["demand"], width=2.5, dash="dash")))
        fig3.add_trace(go.Scatter(x=t, y=cum_aux, name="Auxiliary",
                                  fill="tozeroy", fillcolor="rgba(192,57,43,0.15)",
                                  line=dict(color=C["auxiliary"], width=2)))
        fig3.add_trace(go.Scatter(x=t, y=cum_sol, name="Solar",
                                  fill="tozeroy", fillcolor="rgba(46,204,113,0.25)",
                                  line=dict(color=C["solar_del"], width=2)))
        fig3.update_layout(
            template=TEMPLATE, height=290,
            title="Cumulative Energy",
            xaxis_title="Time (hours)",
            yaxis_title="Energy (kWh)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        # --- Solar fraction over time ---
        sf_pct = h["solar_fraction"] * 100
        fig_sf = go.Figure()
        fig_sf.add_trace(go.Scatter(
            x=t, y=sf_pct,
            fill="tozeroy", fillcolor="rgba(46,204,113,0.2)",
            line=dict(color=C["solar_del"], width=1.5),
            hovertemplate="SF: %{y:.0f}%<extra></extra>",
            showlegend=False,
        ))
        fig_sf.add_hline(
            y=avg_sf,
            line_dash="dash", line_color="navy",
            annotation_text=f"Avg {avg_sf:.0f}%",
            annotation_position="top left",
        )
        fig_sf.update_layout(
            template=TEMPLATE, height=200,
            title="Solar Fraction (%)",
            xaxis_title="Hours",
            yaxis=dict(title="%", range=[0, 105]),
        )
        st.plotly_chart(fig_sf, use_container_width=True)

        # --- Energy split pie ---
        if E_dem > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["Solar", "Auxiliary"],
                values=[max(E_sol, 0), max(E_aux, 0)],
                marker_colors=[C["solar_del"], C["auxiliary"]],
                hole=0.45,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                template=TEMPLATE, height=220,
                title=f"Energy Split<br><sub>Total {E_dem:.1f} kWh</sub>",
                showlegend=False,
                margin=dict(t=55, b=5, l=5, r=5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.metric("Aux Fuel Cost",  f"${cost:.2f}")
        st.metric("Solar Savings",  f"${E_sol * 0.04:.2f}",
                  help="Value of displaced fuel @ $0.04/kWh")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOLAR RADIATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_solar:
    fig_sol = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Irradiance Components (W/m²)",
            "Sun Altitude (°)",
            "Cloud Cover (%)",
        ),
        row_heights=[0.50, 0.30, 0.20],
        vertical_spacing=0.07,
    )

    # Irradiance components
    fig_sol.add_trace(go.Scatter(
        x=t, y=h["irradiance_total"], name="Total",
        fill="tozeroy", fillcolor="rgba(241,196,15,0.15)",
        line=dict(color="#f1c40f", width=2.5),
    ), row=1, col=1)
    fig_sol.add_trace(go.Scatter(
        x=t, y=h["irradiance_direct"], name="Direct",
        line=dict(color=C["direct"], width=1.5),
    ), row=1, col=1)
    fig_sol.add_trace(go.Scatter(
        x=t, y=h["irradiance_diffuse"], name="Diffuse",
        line=dict(color=C["diffuse"], width=1.5),
    ), row=1, col=1)

    # Sun altitude
    alt = h["solar_altitude"]
    fig_sol.add_trace(go.Scatter(
        x=t, y=alt, name="Sun Altitude",
        fill="tozeroy", fillcolor="rgba(255,165,0,0.12)",
        line=dict(color="orange", width=2),
        showlegend=False,
    ), row=2, col=1)
    fig_sol.add_hline(y=0, line_dash="dash", line_color="gray",
                      line_width=1, row=2, col=1)

    # Cloud cover
    fig_sol.add_trace(go.Scatter(
        x=t, y=h["cloud_cover"] * 100, name="Cloud Cover",
        fill="tozeroy", fillcolor="rgba(149,165,166,0.3)",
        line=dict(color=C["diffuse"], width=1.5),
        showlegend=False,
    ), row=3, col=1)

    shade_nights(fig_sol, t, h["irradiance_total"],
                 rows=[1, 2, 3], cols=[1, 1, 1])

    fig_sol.update_xaxes(title_text="Simulation Time (hours)", row=3, col=1)
    fig_sol.update_layout(
        template=TEMPLATE, height=620,
        title="Solar Radiation Analysis",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_sol, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PUMPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pumps:
    fig_p = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Speed (%)", "Electrical Power (W)", "Efficiency (%)"),
        vertical_spacing=0.06,
    )

    pump_series = [
        ("Solar Pump", "solar_pump", C["solar_pump"], "solid"),
        ("Load Pump",  "load_pump",  C["load_pump"],  "dot"),
    ]
    for label, prefix, clr, dash in pump_series:
        kw = dict(color=clr, width=2, dash=dash)
        fig_p.add_trace(go.Scatter(
            x=t, y=h[f"{prefix}_speed"] * 100, name=label,
            line=kw,
            hovertemplate=f"{label} speed: %{{y:.0f}}%<extra></extra>",
        ), row=1, col=1)
        fig_p.add_trace(go.Scatter(
            x=t, y=h[f"{prefix}_power"], name=label, showlegend=False,
            line=kw,
            hovertemplate=f"{label} power: %{{y:.1f}} W<extra></extra>",
        ), row=2, col=1)
        fig_p.add_trace(go.Scatter(
            x=t, y=h[f"{prefix}_efficiency"] * 100, name=label, showlegend=False,
            line=kw,
            hovertemplate=f"{label} η: %{{y:.1f}}%<extra></extra>",
        ), row=3, col=1)

    shade_nights(fig_p, t, h["irradiance_total"],
                 rows=[1, 2, 3], cols=[1, 1, 1])

    fig_p.update_yaxes(range=[-3, 108], row=1, col=1)
    fig_p.update_xaxes(title_text="Simulation Time (hours)", row=3, col=1)
    fig_p.update_layout(
        template=TEMPLATE, height=560,
        title="Pump Performance",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    dt_sec_sim = float(t[1] - t[0]) * 3600 if len(t) > 1 else 60.0
    E_sp = float(np.sum(h["solar_pump_power"])) * dt_sec_sim / 3.6e6  # kWh
    E_lp = float(np.sum(h["load_pump_power"]))  * dt_sec_sim / 3.6e6

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Solar Pump Energy", f"{E_sp * 1000:.1f} Wh")
    pc2.metric("Load Pump Energy",  f"{E_lp * 1000:.1f} Wh")
    pc3.metric("Pump Electricity Cost",
               f"${(E_sp + E_lp) * 0.12:.3f}",
               help="Electricity @ $0.12/kWh")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SOLAR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_sol_exp:
    st.markdown("### Solar Radiation Explorer")
    st.markdown(
        "Adjust sliders to compute the 24-hour irradiance profile "
        "for any day, location, and sky condition in real time."
    )

    xc1, xc2, xc3 = st.columns(3)
    with xc1:
        exp_lat   = st.slider("Latitude (°N)",   25.0,  65.0,  latitude,   0.5, key="xe_lat")
        exp_doy   = st.slider("Day of Year",       1,   365,  start_day,    1,   key="xe_doy")
    with xc2:
        exp_tilt  = st.slider("Panel Tilt (°)",    0,    90,  panel_tilt,   1,   key="xe_tilt")
        exp_cloud = st.slider("Cloud Cover",       0.0,  1.0,  0.1,        0.05, key="xe_cloud",
                              format="%.2f")
    with xc3:
        exp_elev  = st.slider("Elevation (m)",     0,  3000,  elevation,  100,  key="xe_elev")
        exp_lon   = st.slider("Longitude (°E)",  -130.0, -65.0, longitude,  0.5, key="xe_lon")

    exp_loc   = LocationParams(
        latitude =exp_lat,
        longitude=exp_lon,
        timezone =round(exp_lon / 15.0),
        elevation=float(exp_elev),
    )
    exp_solar = SolarRadiationModel(exp_loc)

    hours_fine = np.arange(0, 24, 0.25)
    sol_total, sol_direct, sol_diffuse, sol_alt = [], [], [], []
    for hr in hours_fine:
        s = exp_solar.calculate_irradiance(
            time_hours   =float(hr),
            day_of_year  =int(exp_doy),
            cloud_cover  =float(exp_cloud),
            panel_tilt   =float(exp_tilt),
            panel_azimuth=180.0,
        )
        sol_total.append(s["total"])
        sol_direct.append(s["direct"])
        sol_diffuse.append(s["diffuse"])
        sol_alt.append(s["altitude"])

    sol_total   = np.array(sol_total)
    sol_direct  = np.array(sol_direct)
    sol_diffuse = np.array(sol_diffuse)
    sol_alt     = np.array(sol_alt)

    fig_xe = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
    )
    fig_xe.add_trace(go.Scatter(
        x=hours_fine, y=sol_total, name="Total",
        fill="tozeroy", fillcolor="rgba(241,196,15,0.15)",
        line=dict(color="#f1c40f", width=2.5),
    ), row=1, col=1)
    fig_xe.add_trace(go.Scatter(
        x=hours_fine, y=sol_direct, name="Direct",
        line=dict(color=C["direct"], width=2),
    ), row=1, col=1)
    fig_xe.add_trace(go.Scatter(
        x=hours_fine, y=sol_diffuse, name="Diffuse",
        line=dict(color=C["diffuse"], width=2),
    ), row=1, col=1)
    fig_xe.add_trace(go.Scatter(
        x=hours_fine, y=sol_alt, name="Sun Altitude (°)",
        fill="tozeroy", fillcolor="rgba(255,165,0,0.10)",
        line=dict(color="orange", width=2),
    ), row=2, col=1)
    fig_xe.add_hline(y=0, line_dash="dash", line_color="gray",
                     line_width=1, row=2, col=1)

    fig_xe.update_yaxes(title_text="Irradiance (W/m²)", row=1, col=1)
    fig_xe.update_yaxes(title_text="Altitude (°)", row=2, col=1)
    fig_xe.update_xaxes(
        title_text="Hour of Day", row=2, col=1,
        tickvals=list(range(0, 25, 3)),
        ticktext=[f"{hh:02d}:00" for hh in range(0, 25, 3)],
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    approx_month = month_names[min(int((exp_doy - 1) / 30.4), 11)]
    fig_xe.update_layout(
        template=TEMPLATE, height=480,
        title=(f"Daily Solar Profile — Day {exp_doy} (~{approx_month}), "
               f"{exp_lat:.1f}°N, {exp_cloud*100:.0f}% cloud"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_xe, use_container_width=True)

    peak_irr = float(np.max(sol_total))
    peak_hr  = float(hours_fine[np.argmax(sol_total)]) if peak_irr > 0 else 12.0
    daylight = float(np.sum(sol_alt > 0) * 0.25)
    daily_kwh = float(np.trapz(sol_total, hours_fine) / 1000)  # kWh/m²

    ec1, ec2, ec3, ec4 = st.columns(4)
    ec1.metric("Peak Irradiance",   f"{peak_irr:.0f} W/m²")
    ec2.metric("Peak at",           f"{int(peak_hr):02d}:{int((peak_hr%1)*60):02d}")
    ec3.metric("Daylight Hours",    f"{daylight:.1f} h")
    ec4.metric("Daily Energy",      f"{daily_kwh:.2f} kWh/m²",
               help="Daily insolation on the tilted surface")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PUMP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_pump_exp:
    st.markdown("### Pump H–Q Curve Explorer")
    st.markdown(
        "Visualise how the pump head and efficiency curves shift with speed. "
        "The ★ marks the best-efficiency point (BEP) at each speed."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pe_speed     = st.slider("Highlight Speed (%)", 10, 100, 100, 5, key="pe_spd") / 100.0
        pe_rf_gps    = st.slider("Rated Flow (g/s)",     5, 200,  30,  5, key="pe_rf")
    with pc2:
        pe_rh_m      = st.slider("Rated Head (m)",       1.0, 20.0, 4.0, 0.5, key="pe_rh")
        pe_mh_m      = st.slider("Max Head (m)",         1.0, 30.0, 6.0, 0.5, key="pe_mh")
    with pc3:
        pe_eta_pct   = st.slider("BEP Efficiency (%)",  30, 90, 68,   1, key="pe_eta")

    pump_p = PumpWithCurve(
        "explorer",
        rated_flow    =pe_rf_gps / 1000.0,
        rated_head    =pe_rh_m,
        max_head      =pe_mh_m,
        efficiency_bep=pe_eta_pct / 100.0,
    )

    # H-Q curves at four speeds
    fig_pmp = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Head–Flow (H–Q) Curve", "Efficiency vs Flow Fraction"],
        horizontal_spacing=0.12,
    )

    speed_levels = [
        (1.00, "#e74c3c", 3.0),
        (0.75, "#f39c12", 1.8),
        (0.50, "#3498db", 1.8),
        (0.25, "#95a5a6", 1.8),
    ]
    for sp, clr, lw in speed_levels:
        pump_p.speed = sp
        # sweep flow fraction from 0 → 1.5 (beyond BEP)
        ff_arr  = np.linspace(0, 1.5, 200)
        q_gps   = ff_arr * pump_p.rated_flow * sp * 1000   # g/s on x-axis
        h_arr   = np.array([pump_p.pump_curve(ff) for ff in ff_arr])

        is_hl   = abs(sp - pe_speed) < 1e-3
        opacity = 1.0 if is_hl else 0.45
        fig_pmp.add_trace(go.Scatter(
            x=q_gps, y=h_arr,
            name=f"{int(sp*100)}%",
            line=dict(color=clr, width=lw if is_hl else 1.5),
            opacity=opacity,
            hovertemplate=f"{int(sp*100)}% — Q: %{{x:.1f}} g/s, H: %{{y:.2f}} m<extra></extra>",
        ), row=1, col=1)

        # Mark BEP star on highlighted curve
        if is_hl:
            bep_q = pump_p.rated_flow * sp * 1000
            bep_h = pump_p.pump_curve(1.0)
            fig_pmp.add_trace(go.Scatter(
                x=[bep_q], y=[bep_h],
                name="BEP",
                mode="markers",
                marker=dict(symbol="star", size=16, color="gold",
                            line=dict(color="black", width=1)),
                hovertemplate=f"BEP: Q={bep_q:.1f} g/s, H={bep_h:.2f} m<extra></extra>",
            ), row=1, col=1)

    # Efficiency curve (shape is speed-independent in this model)
    pump_p.speed = 1.0
    ff_range = np.linspace(0, 2.0, 300)
    eta_arr  = np.array([pump_p.efficiency_curve(ff) * 100 for ff in ff_range])
    fig_pmp.add_trace(go.Scatter(
        x=ff_range, y=eta_arr,
        name="Efficiency",
        line=dict(color=C["solar_pump"], width=2.5),
        hovertemplate="FF: %{x:.2f}, η: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)
    # Mark BEP
    fig_pmp.add_trace(go.Scatter(
        x=[1.0], y=[pe_eta_pct],
        name="BEP",
        mode="markers+text",
        marker=dict(symbol="star", size=16, color="gold",
                    line=dict(color="black", width=1)),
        text=[f"  BEP {pe_eta_pct}%"],
        textposition="middle right",
        showlegend=False,
        hovertemplate=f"BEP: η={pe_eta_pct}%<extra></extra>",
    ), row=1, col=2)

    fig_pmp.update_xaxes(title_text="Flow Rate (g/s)",         row=1, col=1)
    fig_pmp.update_xaxes(title_text="Flow Fraction (Q/Q_BEP)", row=1, col=2)
    fig_pmp.update_yaxes(title_text="Head (m)",     row=1, col=1)
    fig_pmp.update_yaxes(title_text="Efficiency (%)", range=[0, 100], row=1, col=2)

    fig_pmp.update_layout(
        template=TEMPLATE, height=420,
        title=f"Pump Performance Curves — Highlighted: {int(pe_speed*100)}% speed",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    st.plotly_chart(fig_pmp, use_container_width=True)

    # Operating point metrics at highlighted speed
    pump_p.speed = pe_speed
    op = pump_p.update(60.0, {"speed": pe_speed})
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Flow Rate",   f"{op['flow_rate']*1000:.1f} g/s")
    oc2.metric("Head",        f"{op['head']:.2f} m")
    oc3.metric("Efficiency",  f"{op['efficiency']*100:.1f}%")
    oc4.metric("Power",       f"{op['power']:.1f} W")

    st.caption(
        "Affinity laws: flow ∝ speed, head ∝ speed², power ∝ speed³. "
        "The quadratic H–Q curve shifts inward at lower speeds. "
        "BEP efficiency is achieved at the rated flow fraction (Q/Q_BEP = 1)."
    )
