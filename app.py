import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    # Dose is set to 1 to allow for normalization later
    return (1.0 * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(curve, target_max=100):
    """Scales any curve to a 0-100 range for visual stability."""
    if np.max(curve) == 0: return curve
    return (curve / np.max(curve)) * target_max

# --- INPUTS ---
# Example: Adderall at 8am, Clonazepam at 2pm
med1_in = input("Enter Primary Drug Date/Time (e.g. 12/24 8am): ")
med2_in = input("Enter Counter Drug Date/Time (e.g. 12/24 2pm): ")

dt_1 = parse_med_time(med1_in)
dt_2 = parse_med_time(med2_in)

# --- ENGINE ---
start_plot = min(dt_1, dt_2) - timedelta(hours=2)
h_axis = np.linspace(0, 36, 1000) 

t_since_1 = np.array([((start_plot + timedelta(hours=h)) - dt_1).total_seconds()/3600 for h in h_axis])
t_since_2 = np.array([((start_plot + timedelta(hours=h)) - dt_2).total_seconds()/3600 for h in h_axis])

# 1. Primary Drug (e.g., Adderall Logic)
# Tmax ~3h, Half-life ~10h
conc_1 = pk_model(t_since_1, 1.1, np.log(2)/10)
conc_1_norm = normalize(conc_1, 100)

# 2. Category 2: REBOUND LOGIC (The "Crash")
# Calculate Velocity of Decline
dC_dt = np.gradient(conc_1_norm, h_axis)
# Trigger: Only when declining AND below 70% of max (the 'Deficit Threshold')
crash_trigger = np.where((dC_dt < 0) & (conc_1_norm < 70), np.abs(dC_dt), 0)
crash_norm = normalize(crash_trigger, 85) # Scaled to 85 so it stays slightly below the main drug

# 3. Counter Drug (e.g., Clonazepam Logic)
# Tmax ~2h, Half-life ~35h
conc_2 = pk_model(t_since_2, 1.8, np.log(2)/35)
conc_2_norm = normalize(conc_2, 100)

# --- VISUALIZATION ---
plt.figure(figsize=(14, 7), facecolor='#fbfbfb')

# Plotting Normalized Curves
plt.plot(h_axis, conc_1_norm, color='#FF8C00', label='Primary Drug (Concentration)', lw=2.5)
plt.plot(h_axis, crash_norm, color='#8E44AD', ls='--', label='Category 2: Rebound/Crash Risk', lw=2)
plt.plot(h_axis, conc_2_norm, color='#27AE60', label='Counter Drug (Relief)', lw=3)

# The Mitigation Area
# Where the Relief (Green) meets the Rebound (Purple)
mitigation = np.minimum(crash_norm, conc_2_norm)
plt.fill_between(h_axis, 0, mitigation, color='#F1C40F', alpha=0.4, label='Neutralization Window')

# Formatting
tick_indices = np.arange(0, 37, 4)
plt.xticks(tick_indices, [(start_plot + timedelta(hours=int(h))).strftime("%m/%d\n%I%p") for h in tick_indices])

plt.title("Visual Decision Support: Normalized PK/PD Interaction Map", fontsize=15, fontweight='bold')
plt.xlabel("Timeline (Calendar Time)", fontsize=12)
plt.ylabel("Intensity / Saturation (%)", fontsize=12) # Standardized Y-axis
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(alpha=0.2, linestyle=':')
plt.axhline(0, color='black', lw=1)

plt.tight_layout()
plt.show()
