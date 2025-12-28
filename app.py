import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. CORE ENGINE: PHARMACOKINETICS ---
def pk_model(t, dose, ka, ke):
    """Standard 1-Compartment PK Model"""
    t = np.maximum(t, 0)
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

# --- 2. DATASET: ABILIFY & COUNTER-MEDS ---
# Based on FDA Labeling Data
ABILIFY_DATA = {
    "ka": 1.0, 
    "ke": np.log(2) / 75, # 75h half-life
    "dose": 100,
    "side_effects": {
        "Akathisia (Restlessness)": {"lag": 2.0, "color": "red", "type": "neurological"},
        "Insomnia": {"lag": 4.0, "color": "orange", "type": "metabolic"},
        "Nausea": {"lag": 0.5, "color": "magenta", "type": "direct"},
        "Dizziness": {"lag": 1.0, "color": "brown", "type": "direct"},
        "Fatigue/Somnolence": {"lag": 5.0, "color": "purple", "type": "metabolic"},
        "Blurred Vision": {"lag": 2.0, "color": "gray", "type": "direct"},
        "Anxiety": {"lag": 1.5, "color": "pink", "type": "neurological"},
        "Tremor": {"lag": 2.5, "color": "darkred", "type": "neurological"},
        "Dry Mouth": {"lag": 3.0, "color": "cyan", "type": "direct"},
        "Headache": {"lag": 2.0, "color": "olive", "type": "direct"}
    }
}

COUNTER_MEDS = {
    "None": None,
    "Clonazepam": {"ka": 1.8, "ke": np.log(2) / 35, "t_max": 2.0},
    "Propranolol": {"ka": 2.5, "ke": np.log(2) / 4.0, "t_max": 1.5},
    "Melatonin": {"ka": 3.0, "ke": np.log(2) / 1.0, "t_max": 0.5},
    "Guanfacine": {"ka": 1.2, "ke": np.log(2) / 17, "t_max": 3.0},
    "Hydroxyzine": {"ka": 2.0, "ke": np.log(2) / 20, "t_max": 2.0},
    "Benadryl": {"ka": 2.2, "ke": np.log(2) / 8, "t_max": 2.0},
    "Zofran": {"ka": 2.8, "ke": np.log(2) / 3.5, "t_max": 1.5},
    "Trazodone": {"ka": 1.5, "ke": np.log(2) / 10, "t_max": 2.0},
    "Magnesium": {"ka": 0.8, "ke": np.log(2) / 12, "t_max": 4.0},
    "Xanax": {"ka": 3.0, "ke": np.log(2) / 11, "t_max": 1.0}
}

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Med-Engineering Dashboard", layout="wide")
st.title("💊 Multi-Million Dollar Idea: Med-Engineering Dashboard")
st.subheader("Pilot Prototype: Abilify (Aripiprazole) Optimization")

with st.sidebar:
    st.header("1. Primary Medication")
    med = st.selectbox("Select Medicine", ["Abilify"])
    
    # Date/Time Input
    st.header("2. Timing & Date")
    dose_date = st.date_input("Dose Date", datetime(2025, 12, 9))
    dose_time = st.time_input("Dose Time", value=datetime.strptime("10:00 PM", "%I:%M %p").time())
    
    # Side Effect Selection
    st.header("3. Symptom Mapping")
    selected_se = st.selectbox("Select Side Effect to Mitigate", list(ABILIFY_DATA["side_effects"].keys()))
    
    # Counter Med Selection
    st.header("4. Counter-Medication")
    counter_med = st.selectbox("Select Counter Medication", list(COUNTER_MEDS.keys()))

# --- 4. CALCULATION ---
dt_dose = datetime.combine(dose_date, dose_time)
t_plot = np.linspace(0, 48, 1000) # 48 hour view

# Primary Curve
abilify_conc = pk_model(t_plot, ABILIFY_DATA["dose"], ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

# Side Effect Curve (Custom Logic Kernel)
se_lag = ABILIFY_DATA["side_effects"][selected_se]["lag"]
se_color = ABILIFY_DATA["side_effects"][selected_se]["color"]
side_effect_conc = pk_model(t_plot - se_lag, ABILIFY_DATA["dose"] * 0.8, ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

# Recommender Logic
peak_hour = t_plot[np.argmax(side_effect_conc)]
predicted_peak_time = dt_dose + timedelta(hours=peak_hour)

# Counter Med Curve
counter_conc = np.zeros_like(t_plot)
rec_time_str = "N/A"
if counter_med != "None":
    c_data = COUNTER_MEDS[counter_med]
    # Optimal logic: Peak of counter med should hit at peak of side effect
    optimal_offset = peak_hour - c_data["t_max"]
    counter_conc = pk_model(t_plot - optimal_offset, 110, c_data["ka"], c_data["ke"])
    rec_time = dt_dose + timedelta(hours=optimal_offset)
    rec_time_str = rec_time.strftime("%m/%d %I:%M %p")

# --- 5. SMART RECOMMENDER DISPLAY ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Side Effect Peak Predicted", predicted_peak_time.strftime("%m/%d %I:%M %p"))
with col2:
    st.metric("Recommended Counter-Med Time", rec_time_str)

# --- 6. PLOT ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_plot, abilify_conc, label="Abilify (Blood Conc.)", color="blue", alpha=0.3)
ax.plot(t_plot, side_effect_conc, label=f"RISK: {selected_se}", color=se_color, lw=3)

if counter_med != "None":
    ax.plot(t_plot, counter_conc, label=f"COUNTER: {counter_med}", color="green", lw=3)
    # Highlight Mitigation
    mitigation = np.minimum(side_effect_conc, counter_conc)
    ax.fill_between(t_plot, 0, mitigation, color="gold", alpha=0.3, label="Mitigation Zone")

ax.set_xlabel("Hours since start")
ax.set_ylabel("Intensity")
ax.legend()
st.pyplot(fig)
