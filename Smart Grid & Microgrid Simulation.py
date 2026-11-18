"""
ECE181 First-Semester Python Project: Smart Grid & Microgrid Simulation
Author: ECE Student
Course: ECE181 Introduction to Python (30 Marks Assessment)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set visual styling for publication-quality plots

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})



# 1. EXTERNAL DATA LOADING & DEFENSIVE CLEANING

def generate_and_clean_dataset(filename: str = "smart_grid_data.csv") -> pd.DataFrame:
    """Loads 24-hour smart meter load and solar logs from an external CSV file,
    applies defensive cleaning, and calculates unmitigated net grid power.
    """
    # Create sample CSV file if it doesn't exist to ensure seamless execution

    if not os.path.exists(filename):
        hours = np.arange(24)
        raw_load = np.array([
            70.5, 68.8, 72.2, 78.5, 85.1, 95.7, 110.3, 115.8, 120.1, 125.2,
            115.0, 105.5, 95.2, 90.8, 91.1, 95.0, 110.6, 140.0, 175.0, 165.0,
            145.0, 115.0, 88.0, 75.8
        ])
        # Adjusted solar yield to exceed load at midday, producing negative net power (surplus)

        raw_solar = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 20.0, 50.0, 90.0, 130.0,
            150.0, 160.0, 160.0, 140.0, 110.0, 80.0, 45.0, 15.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ])
        sample_df = pd.DataFrame({
            "Hour": hours,
            "Consumer_Load_kW": raw_load,
            "Solar_PV_kW": raw_solar
        })
        sample_df.to_csv(filename, index=False)

    # Load real-world dataset via pandas read_csv

    df = pd.read_csv(filename)

    # Defensive cleaning: Impute negative/null spikes using .clip() and .ffill/.fillna

    df["Consumer_Load_kW"] = df["Consumer_Load_kW"].clip(lower=0.0).ffill()
    df["Solar_PV_kW"] = df["Solar_PV_kW"].clip(lower=0.0).fillna(0.0)

    # Calculate unmitigated net grid power: P_net(t) = P_load(t) - P_solar(t)

    df["Net_Power_kW"] = df["Consumer_Load_kW"] - df["Solar_PV_kW"]

    return df



# 2. DISCRETE TIME-STEP SIMULATION ENGINE

def run_microgrid_simulation(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Executes discrete time-step energy balance modeling across 24 hours."""
    sim_df = df.copy()

    n_steps = len(sim_df)
    soc = np.zeros(n_steps)
    p_charge = np.zeros(n_steps)
    p_discharge = np.zeros(n_steps)
    mitigated_grid_load = np.zeros(n_steps)

    # Initial condition: 35% State of Charge (SoC) at t=0

    current_energy = (35.0 / 100.0) * params["E_capacity"]

    for t in range(n_steps):
        p_net = sim_df.loc[t, "Net_Power_kW"]

        # Case A: Surplus Solar Overproduction (P_net < 0) -> Charge Battery

        if p_net < 0:
            surplus = abs(p_net)
            rem_cap = params["E_capacity"] - current_energy
            max_charge_possible = rem_cap / params["eff_c"]

            charge_kw = min(surplus, params["P_max_rate"], max_charge_possible)
            p_charge[t] = charge_kw
            p_discharge[t] = 0.0

            current_energy += charge_kw * params["eff_c"]
            mitigated_grid_load[t] = 0.0

        # Case B: Peak Demand Exceeds Target Cap (P_net > P_threshold) -> Discharge

        elif p_net > params["P_threshold"]:
            deficit = p_net - params["P_threshold"]
            available_discharge = current_energy * params["eff_d"]

            discharge_kw = min(deficit, params["P_max_rate"], available_discharge)
            p_discharge[t] = discharge_kw
            p_charge[t] = 0.0

            current_energy -= discharge_kw / params["eff_d"]
            mitigated_grid_load[t] = p_net - discharge_kw

        # Case C: Passive Grid Operation

        else:
            p_charge[t] = 0.0
            p_discharge[t] = 0.0
            mitigated_grid_load[t] = max(0.0, p_net)

        # Update bounded State of Charge (%)

        soc[t] = (current_energy / params["E_capacity"]) * 100.0

    sim_df["P_Charge_kW"] = p_charge
    sim_df["P_Discharge_kW"] = p_discharge
    sim_df["Mitigated_Grid_Load_kW"] = mitigated_grid_load
    sim_df["Battery_SoC_Pct"] = soc

    return sim_df



# 3. MULTI-SCENARIO EXECUTION PIPELINE

def run_multi_scenario_analysis(df: pd.DataFrame) -> dict:
    """Runs multiple simulation scenarios comparing different battery capacities."""
    scenarios = {
        "Small Battery (100 kWh)": {
            "E_capacity": 100.0,
            "P_max_rate": 30.0,
            "eff_c": 0.92,
            "eff_d": 0.92,
            "P_threshold": 50.0,
        },
        "Standard Battery (200 kWh)": {
            "E_capacity": 200.0,
            "P_max_rate": 40.0,
            "eff_c": 0.92,
            "eff_d": 0.92,
            "P_threshold": 50.0,
        },
        "Large Battery (300 kWh)": {
            "E_capacity": 300.0,
            "P_max_rate": 60.0,
            "eff_c": 0.92,
            "eff_d": 0.92,
            "P_threshold": 50.0,
        },
    }

    results = {}
    for name, params in scenarios.items():
        results[name] = run_microgrid_simulation(df, params)
    return results



# 4. COMPARATIVE VISUALIZATION & METRICS

def plot_comparative_results(results: dict, df_raw: pd.DataFrame):
    """Generates comparative multi-scenario plots for grid load and battery SoC."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    # Panel 1: Comparative Net Grid Load

    axes[0].plot(df_raw["Hour"], df_raw["Net_Power_kW"], 'k--', label="Unmitigated Net Load", linewidth=2, alpha=0.6)
    for name, sim_df in results.items():
        axes[0].plot(sim_df["Hour"], sim_df["Mitigated_Grid_Load_kW"], label=name, linewidth=2)
    axes[0].axhline(y=50.0, color='r', linestyle=':', label="Target Cap (50 kW)")
    axes[0].set_title("Multi-Scenario Peak-Shaving Comparison", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hour of Day (0 - 23)")
    axes[0].set_ylabel("Net Power (kW)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Comparative Battery State of Charge (%)

    for name, sim_df in results.items():
        axes[1].plot(sim_df["Hour"], sim_df["Battery_SoC_Pct"], label=name, linewidth=2)
    axes[1].set_title("Battery State of Charge Across Scenarios", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Hour of Day (0 - 23)")
    axes[1].set_ylabel("State of Charge (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Execution Pipeline

if __name__ == "__main__":
    df_dataset = generate_and_clean_dataset("smart_grid_data.csv")
    scenario_results = run_multi_scenario_analysis(df_dataset)
    plot_comparative_results(scenario_results, df_dataset)