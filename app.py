import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- STREAMLIT UI CONFIG ---
st.set_page_config(page_title="Medication Timing Optimizer", layout="wide")
st.title("💊 Medication Visual Engineering SaaS")
st.markdown("""
This tool visualizes **Pharmacokinetic (PK)** interactions and **Category 2 (Rebound)** symptoms 
to help optimize dosage timing.
""")

# --- HELPER FUNCTIONS ---
def parse_med_time(dt_str):
    """Parses 'mm/dd time' (e.g., '12/24 10pm')."""
    dt_str = dt_str.lower().strip()
    full_str = f"{dt_str} 2025" 
    formats = ["%m/%d %I%p %Y", "%m/%d %I:%M%p %Y", "%m/%d %H:%M %Y"]
    for fmt in formats:
        try: return datetime.strptime(full_str, fmt)
        except ValueError: continue
    return None

def pk_model(t_hours, ka, ke):
    """Standard 1-compartment PK model (Unit Dose)."""
    t = np.maximum(t_hours, 0)
    return (1.0 * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(curve, target_max=100):
    """Scales any curve to a 0-100 range for visual stability."""
    if np.max(curve) == 0: return curve
    return (curve / np.max(curve)) * target_max

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Inputs")
abilify_in = st.sidebar.text_input("Primary Drug (e.g. Adderall) MM/DD Time:", "12/24 8am")
clon_in = st.sidebar.text_input("Counter Drug (e.g. Clonazepam) MM/DD Time:", "12/24 2pm")

dt_1 = parse_med_time(abilify_in)
dt_2 = parse_med_time(clon_in)

if dt_1 and dt_2:
    # --- ENGINE ---
    start_plot = min(dt_1, dt_2) - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000) 

    t_since_1 = np.array([((start_plot + timedelta(hours=h)) - dt_1).total_seconds()/3600 for h in h_axis])
    t_since_2 = np.array([((start_plot + timedelta(hours=h)) - dt_2).total_seconds()/3600 for h in h_axis])

    # 1. Primary Drug Logic (e.g., Adderall)
    # Tmax ~3h, Half-life ~10h
    conc_1 = pk_model(t_since_1, 1.1, np.log(2)/10)
    conc_1_norm = normalize(conc_1, 100)

    # 2. Category 2: REBOUND LOGIC (The "Crash")
    # UPDATED: Trigger only when declining AND concentration < 50% of peak
    dC_dt = np.gradient(conc_1_norm, h_axis)
    crash_trigger = np.where((dC_dt < 0) & (conc_1_norm < 50), np.abs(dC_dt), 0)
    crash_norm = normalize(crash_trigger, 85) 

    # 3. Counter Drug Logic (e.g., Clonazepam)
    conc_2 = pk_model(t_since_2, 1.8, np.log(2)/35)
    conc_2_norm = normalize(conc_2, 100)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#fbfbfb')

    # Plotting
    ax.plot(h_axis, conc_1_norm, color='#FF8C00', label='Primary Drug (Conc.)', lw=2.5)
    ax.plot(h_axis, crash_norm, color='#8E44AD', ls='--', label='Category 2: Crash (Starts at <50% Conc.)', lw=2)
    ax.plot(h_axis, conc_2_norm, color='#27AE60', label='Counter Drug (Relief)', lw=3)

    # The Neutralization Window
    mitigation = np.minimum(crash_norm, conc_2_norm)
    ax.fill_between(h_axis, 0, mitigation, color='#F1C40F', alpha=0.4, label='Neutralization Window')

    # X-Axis Formatting
    tick_indices = np.arange(0, 37, 4)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d\n%I%p") for h in tick_indices])

    ax.set_title("Clinical Decision Support: Timing Optimization", fontsize=14)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Normalized Intensity (%)")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2, linestyle=':')

    st.pyplot(fig)

    # --- ACTIONABLE INSIGHT ---
    st.subheader("Analysis & Recommendations")
    max_mitigation = np.max(mitigation)
    if max_mitigation > 40:
        st.success(f"Optimal Timing Detected! The counter-drug successfully neutralizes {int(max_mitigation)}% of the crash intensity.")
    else:
        st.warning("Sub-optimal Timing: The relief peak does not align with the crash peak. Consider shifting the counter-drug dose time.")

else:
    st.error("Please enter a valid date and time in the sidebar (MM/DD HHam/pm).")
