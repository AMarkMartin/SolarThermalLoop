"""
Generate clean, presentation-quality individual plots from the winter scenario.

Produces separate figures optimized for embedding in documentation and slides:
  1. system_temperatures.png  — Tank, collector, outdoor temps over 5 days
  2. energy_breakdown.png     — Stacked area: solar vs auxiliary heating
  3. cumulative_energy.png    — Cumulative demand, solar, and auxiliary
  4. solar_fraction.png       — Instantaneous solar fraction with average line
  5. control_signals.png      — Pump speed and solar irradiance overlay

Run from project root:
    uv run examples/generate_presentation_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

from src.components import (
    SolarCollector, SolarCollectorParams,
    StorageTank, StorageTankParams,
    Component, PumpWithCurve
)
from src.models import (
    SolarRadiationModel, LocationParams, WeatherForecast
)
from src.control import BasicController

# Re-use the winter scenario system
from examples.winter_scenario import SimplifiedBuildingLoad, WinterHeatingSystem

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_simulation():
    """Run a spring shoulder-season scenario and return the system with populated history."""
    location = LocationParams(
        latitude=40.0, longitude=-105.0, timezone=-7.0, elevation=1600.0
    )
    collector = SolarCollector(
        "SolarCollector",
        SolarCollectorParams(area=15.0, efficiency=0.75, heat_loss_coef=4.0)
    )
    tank = StorageTank(
        "StorageTank",
        StorageTankParams(volume=0.8, mass=800.0, surface_area=4.5, heat_loss_coef=1.0),
        initial_temp=35.0
    )
    building = SimplifiedBuildingLoad(
        "ResidentialBuilding",
        floor_area=200.0,
        design_heat_load=5000.0,   # 5 kW at design conditions
        design_outdoor_temp=-10.0,
        indoor_setpoint=21.0
    )
    controller = BasicController(solar_dT_threshold=5.0, tank_max_temp=75.0)
    system = WinterHeatingSystem(
        collector=collector, tank=tank, building=building,
        controller=controller, location=location,
        start_day=75, base_temp=5.0  # Early March — shoulder season
    )
    system.run_simulation(duration_hours=120.0, dt=60.0)
    return system


def day_formatter(ax, label='Time (days)'):
    """Format x-axis as days instead of hours."""
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/24:.0f}'))
    ax.set_xlabel(label)


def plot_system_temperatures(system):
    """Plot 1: System temperatures — tank, collector, outdoor."""
    h = system.history
    time = np.array(h['time'])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, h['T_tank'], color='#d62728', linewidth=2.5, label='Storage Tank')
    ax.plot(time, h['T_collector'], color='#ff7f0e', linewidth=1.5, alpha=0.8, label='Collector')
    ax.plot(time, h['T_outdoor'], color='#1f77b4', linewidth=1.5, alpha=0.7, label='Outdoor')
    ax.axhline(y=0, color='lightblue', linestyle=':', linewidth=1, alpha=0.5)

    # Shade nighttime
    for day in range(5):
        ax.axvspan(day * 24 + 18, day * 24 + 30, color='#dde', alpha=0.3)

    ax.set_ylabel('Temperature (°C)')
    ax.set_title('System Temperatures — 5-Day Spring Scenario')
    ax.legend(loc='upper right', framealpha=0.9)
    day_formatter(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'system_temperatures.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved system_temperatures.png')


def plot_energy_breakdown(system):
    """Plot 2: Heating breakdown — solar vs auxiliary (stacked area)."""
    h = system.history
    time = np.array(h['time'])
    Q_solar = np.array(h['Q_solar_delivered']) / 1000  # kW
    Q_aux = np.array(h['Q_auxiliary']) / 1000
    Q_demand = np.array(h['Q_building_demand']) / 1000

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(time, 0, Q_solar, color='#2ca02c', alpha=0.6, label='Solar Delivered')
    ax.fill_between(time, Q_solar, Q_solar + Q_aux, color='#d62728', alpha=0.4, label='Auxiliary (Gas)')
    ax.plot(time, Q_demand, 'k--', linewidth=1.5, label='Total Demand')

    ax.set_ylabel('Heating Power (kW)')
    ax.set_title('Building Heating Breakdown')
    ax.legend(loc='upper right', framealpha=0.9)
    day_formatter(ax)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'energy_breakdown.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved energy_breakdown.png')


def plot_cumulative_energy(system):
    """Plot 3: Cumulative energy — demand, solar contribution, auxiliary."""
    h = system.history
    time = np.array(h['time'])
    dt_hours = time[1] - time[0] if len(time) > 1 else 1 / 60

    cum_demand = np.cumsum(h['Q_building_demand']) * dt_hours / 1000  # kWh
    cum_solar = np.cumsum(h['Q_solar_delivered']) * dt_hours / 1000
    cum_aux = np.cumsum(h['Q_auxiliary']) * dt_hours / 1000

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(time, cum_demand, 'k--', linewidth=2.5, label=f'Total Demand ({cum_demand[-1]:.0f} kWh)')
    ax.fill_between(time, 0, cum_solar, color='#2ca02c', alpha=0.35)
    ax.plot(time, cum_solar, color='#2ca02c', linewidth=2.5, label=f'Solar ({cum_solar[-1]:.0f} kWh)')
    ax.fill_between(time, cum_solar, cum_demand, color='#d62728', alpha=0.15)
    ax.plot(time, cum_aux, color='#d62728', linewidth=2, label=f'Auxiliary ({cum_aux[-1]:.0f} kWh)')

    ax.set_ylabel('Cumulative Energy (kWh)')
    ax.set_title('Cumulative Energy Balance')
    ax.legend(loc='upper left', framealpha=0.9)
    day_formatter(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'cumulative_energy.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved cumulative_energy.png')


def plot_solar_fraction(system):
    """Plot 4: Instantaneous solar fraction."""
    h = system.history
    time = np.array(h['time'])
    sf = np.array(h['solar_fraction']) * 100

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(time, 0, sf, color='#2ca02c', alpha=0.4)
    ax.plot(time, sf, color='#2ca02c', linewidth=1.5)
    avg = np.mean(sf)
    ax.axhline(y=avg, color='#1f77b4', linestyle='--', linewidth=2,
               label=f'Average: {avg:.1f}%')
    ax.set_ylabel('Solar Fraction (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Instantaneous Solar Fraction')
    ax.legend(loc='upper right', framealpha=0.9)
    day_formatter(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'solar_fraction.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved solar_fraction.png')


def plot_control_signals(system):
    """Plot 5: Pump control overlaid with solar irradiance."""
    h = system.history
    time = np.array(h['time'])

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Irradiance on secondary axis
    ax2 = ax1.twinx()
    ax2.fill_between(time, 0, h['irradiance_total'], color='#ffcc00', alpha=0.3, label='Irradiance')
    ax2.set_ylabel('Irradiance (W/m²)', color='#b08800')
    ax2.tick_params(axis='y', labelcolor='#b08800')
    ax2.set_ylim(0, 600)

    # Pump speeds
    ax1.plot(time, np.array(h['solar_pump_speed']) * 100,
             color='#2ca02c', linewidth=2, label='Solar Pump', zorder=5)
    ax1.plot(time, np.array(h['load_pump_speed']) * 100,
             color='#1f77b4', linewidth=2, label='Load Pump', zorder=5)
    ax1.set_ylabel('Pump Speed (%)')
    ax1.set_ylim(-5, 110)
    ax1.set_title('Pump Control Signals & Solar Irradiance')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

    day_formatter(ax1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'control_signals.png'), bbox_inches='tight')
    plt.close(fig)
    print('  Saved control_signals.png')


def main():
    print('Running winter scenario simulation...')
    system = run_simulation()

    print('\nGenerating presentation plots:')
    plot_system_temperatures(system)
    plot_energy_breakdown(system)
    plot_cumulative_energy(system)
    plot_solar_fraction(system)
    plot_control_signals(system)

    print(f'\nAll plots saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
