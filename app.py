import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# --- 1. CORE ENGINE (The Secret Sauce) ---

def pk_model(t, dose, ka, ke):
    """Standard 1-compartment PK model normalized to 0-100 scale."""
    t = np.maximum(t, 0)
    curve = (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    return curve

def get_se_curve(chief_curve, category, t_axis):
    """Generates Side Effect profiles based on Category logic."""
    if category == "Category 1 (Trailing)":
        # Shift curve by 1.5 hours (Lag)
        return np.roll(chief_curve, 15) 
    
    elif category == "Category 2 (Rebound/Crash)":
        # Logic: Starts after 50% dip from peak AND negative slope
        peak_val = np.max(chief_curve)
        peak_idx = np.argmax(chief_curve)
        
        # Calculate slope
        slope = np.gradient(chief_curve)
        
        crash = np.zeros_like(chief_curve)
        for i in range(len(chief_curve)):
            if i > peak_idx and chief_curve[i] < (0.5 * peak_val) and slope[i] < 0:
                # Intensity is proportional to the steepness of the drop
                crash[i] = np.abs(slope[i]) * 10 
        
        # Normalize the crash for visual distinctness
        if np.max(crash) > 0:
            crash = (crash / np.max(crash)) * 80
        return crash
    
    return np.zeros_like(chief_curve)

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Med-Engineer SaaS", layout="wide")
st.title("💊 Clinical Timing Optimizer (SaaS MVP)")
st.markdown("### Precision Side-Effect Mitigation Engine")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Patient Regimen")
    
    # Drug 1: The Primary
    st.subheader("Primary Medication")
    drug_a_name = st.selectbox("Select Drug", ["Abilify", "Adderall IR", "Lexapro"])
    date_a = st.date_input("Date (Drug A)", datetime(2025, 12, 21))
    time_a = st.time_input("Time (Drug A)", value=datetime.strptime("08:00", "%H:%M"))
    
    # Drug 2: The Counter-Med
    st.subheader("Counter Medication")
    drug_b_name = st.selectbox("Select Counter-Drug", ["Clonazepam", "Xanax", "Propranolol"])
    date_b = st.date_input("Date (Drug B)", datetime(2025, 12, 21))
    time_b = st.time_input("Time (Drug B)", value=datetime.strptime("14:00", "%H:%M"))

    st.header("Side Effect Profile")
    se_type = st.radio("Side Effect Type", ["Category 1 (Trailing)", "Category 2 (Rebound/Crash)"])

# --- 3. DATA PROCESSING ---

# Define Constants (Approximated)
# [ka, ke, t_max_approx]
params = {
    "Abilify": [1.0, 0.009, 4], 
    "Adderall IR": [1.1, 0.069, 3],
    "Lexapro": [0.8, 0.023, 5],
    "Clonazepam": [1.8, 0.019, 2],
    "Xanax": [1.5, 0.05, 1],
    "Propranolol": [1.2, 0.17, 1.5]
}

# Setup Timeline
start_dt = datetime.combine(date_a, time_a)
dt_a = start_dt
dt_b = datetime.combine(date_b, time_b)

h_axis = np.linspace(0, 36, 1000)
t_since_a = np.array([((start_dt + timedelta(hours=h)) - dt_a).total_seconds()/3600 for h in h_axis])
t_since_b = np.array([((start_dt + timedelta(hours=h)) - dt_b).total_seconds()/3600 for h in h_axis])

# Calculate Curves
p_a = params[drug_a_name]
p_b = params[drug_b_name]

chief_a = pk_model(t_since_a, 100, p_a[0], p_a[1])
# Normalize Chief A
chief_a = (chief_a / np.max(chief_a)) * 100

side_effect = get_se_curve(chief_a, se_type, h_axis)

chief_b = pk_model(t_since_b, 100, p_b[0], p_b[1])
# Normalize Chief B to be distinct but balanced
chief_b = (chief_b / np.max(chief_b)) * 90 if np.max(chief_b) > 0 else chief_b

# Calculate Mitigation Percentage
mitigation_area = np.minimum(side_effect, chief_b)
total_se_area = np.sum(side_effect)
mitigation_pct = (np.sum(mitigation_area) / total_se_area * 100) if total_se_area > 0 else 0

# --- 4. VISUALIZATION ---

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(h_axis, chief_a, color='#3498db', label=f'{drug_a_name} (Conc.)', alpha=0.5)
ax.plot(h_axis, side_effect, color='#e74c3c', ls='--', label=f'Predicted {se_type}', lw=2)
ax.plot(h_axis, chief_b, color='#2ecc71', label=f'{drug_b_name} (Relief)', lw=3)

ax.fill_between(h_axis, 0, mitigation_area, color='#f1c40f', alpha=0.4, label='Symptom Mitigation Zone')

# Formatting
ax.set_title(f"Medication Engineering: {drug_a_name} vs. {se_type}", fontsize=14)
ax.set_xlabel("Hours Since First Dose")
ax.set_ylabel("Normalized Intensity (%)")
ax.set_ylim(0, 110)
ax.grid(alpha=0.2)
ax.legend()

# Display in Streamlit
st.pyplot(fig)

# --- 5. OUT-OF-THE-BOX IP FEATURES ---

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mitigation Score", f"{int(mitigation_pct)}%")
    st.caption("How much of the side effect is covered by the counter-med.")

with col2:
    risk_level = "High" if mitigation_pct < 40 else "Low"
    st.metric("Patient Vulnerability", risk_level)
    st.caption("Risk of dropout due to side effects.")

with col3:
    # Logic to suggest better timing
    best_time = "Shift counter-med by +2 hours" if mitigation_pct < 60 else "Timing Optimal"
    st.metric("AI Recommendation", best_time)
    st.caption("Suggested timing adjustment.")

st.info("**IP Note:** This model uses Rebound Velocity logic for Category 2 symptoms, triggering only after a 50% decline in plasma levels.")
