import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. THE DATA REGISTRY (The "Sauce") ---
DRUG_DB = {
    "Abilify (Aripiprazole)": {"ka": 1.0, "ke": np.log(2)/75, "side_effects": {"Akathisia": 1, "Insomnia": 1, "Crash": 2}},
    "Adderall IR": {"ka": 1.1, "ke": np.log(2)/10, "side_effects": {"Anxiety": 1, "Crash": 2}},
    "Lexapro (Escitalopram)": {"ka": 0.8, "ke": np.log(2)/30, "side_effects": {"Nausea": 1, "Fatigue": 1}}
}

COUNTER_DB = {
    "Clonazepam": {"ka": 1.8, "ke": np.log(2)/35},
    "Propranolol": {"ka": 2.0, "ke": np.log(2)/4},
    "L-Theanine": {"ka": 1.5, "ke": np.log(2)/3}
}

# --- 2. THE ENGINE FUNCTIONS ---
def pk_model(t, dose, drug_data):
    t = np.maximum(t, 0)
    ka, ke = drug_data["ka"], drug_data["ke"]
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def get_se_intensity(chief, se_name, drug_name):
    logic_type = DRUG_DB[drug_name]["side_effects"][se_name]
    if logic_type == 1: # Trailing
        return np.roll(chief, 15) # 1.5h shift
    if logic_type == 2: # Rebound (Velocity)
        grad = np.gradient(chief)
        return np.where(grad < 0, np.abs(grad) * 120, 0)
    return np.zeros_like(chief)

# --- 3. THE UI BUILDER ---
st.set_page_config(page_title="Neuro-Optimizer CDSS", layout="wide")
st.title("🧠 Neuro-Optimizer: Clinical Decision Support")
st.markdown("---")

# Sidebar for Doctor Inputs
with st.sidebar:
    st.header("📋 Prescription Setup")
    
    # 1. Primary Medication
    selected_drug = st.selectbox("Select Medication", list(DRUG_DB.keys()))
    d_date = st.date_input("Start Date")
    d_time = st.time_input("Dose Time")
    
    # 2. Side Effect Selection
    se_list = list(DRUG_DB[selected_drug]["side_effects"].keys())
    selected_se = st.selectbox("Monitor Side Effect", se_list)
    
    st.markdown("---")
    
    # 3. Counter Medication
    selected_counter = st.selectbox("Counter Medication", ["None"] + list(COUNTER_DB.keys()))
    c_time = st.time_input("Counter Dose Time", value=(datetime.combine(d_date, d_time) + timedelta(hours=4)).time())

# --- 4. CALCULATION & VISUALIZATION ---
h_axis = np.linspace(0, 36, 1000) 
dose_dt = datetime.combine(d_date, d_time)
counter_dt = datetime.combine(d_date, c_time)

# Calculate offset in hours
offset = (counter_dt - dose_dt).total_seconds() / 3600

# Generate Curves
chief_curve = pk_model(h_axis, 100, DRUG_DB[selected_drug])
se_curve = get_se_intensity(chief_curve, selected_se, selected_drug)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(h_axis, chief_curve, label="Chief Concentration", color="blue", alpha=0.3)
ax.plot(h_axis, se_curve, label=f"Risk: {selected_se}", color="red", ls="--", lw=2)

if selected_counter != "None":
    counter_curve = pk_model(h_axis - offset, 110, COUNTER_DB[selected_counter])
    ax.plot(h_axis, counter_curve, label=f"Relief: {selected_counter}", color="green", lw=2.5)
    
    # Show Mitigation
    mitigation = np.minimum(se_curve, counter_curve)
    ax.fill_between(h_axis, 0, mitigation, color="gold", alpha=0.4, label="Mitigation Zone")
    
    # Metrics
    score = int(np.sum(mitigation) / np.sum(se_curve) * 100) if np.sum(se_curve) > 0 else 0
    st.metric("Mitigation Score", f"{score}%")

ax.set_title(f"24-Hour Forecast: {selected_se} Management")
ax.legend()
st.pyplot(fig)

st.info("💡 Tip: Adjust the 'Counter Dose Time' in the sidebar to maximize the Gold Mitigation Zone.")
