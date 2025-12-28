import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. CORE ENGINE: PHARMACOKINETICS ---
def pk_model(t, dose, ka, ke):
    t = np.maximum(t, 0)
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

# --- 2. DATASET ---
ABILIFY_DATA = {
    "ka": 1.0, "ke": np.log(2) / 75, "dose": 100,
    "side_effects": {
        "Akathisia (Restlessness)": {"lag": 2.0, "color": "#e74c3c", "type": "neurological"},
        "Insomnia": {"lag": 4.0, "color": "#e67e22", "type": "metabolic"},
        "Nausea": {"lag": 0.5, "color": "#9b59b6", "type": "direct"},
        "Dizziness": {"lag": 1.0, "color": "#a04000", "type": "direct"},
        "Fatigue/Somnolence": {"lag": 5.0, "color": "#8e44ad", "type": "metabolic"},
        "Blurred Vision": {"lag": 2.0, "color": "#7f8c8d", "type": "direct"},
        "Anxiety": {"lag": 1.5, "color": "#f06292", "type": "neurological"},
        "Tremor": {"lag": 2.5, "color": "#c0392b", "type": "neurological"},
        "Dry Mouth": {"lag": 3.0, "color": "#1abc9c", "type": "direct"},
        "Headache": {"lag": 2.0, "color": "#273746", "type": "direct"}
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

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Med-Engineering v2", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Chronopharmacology Dashboard")
st.info("Precision Side-Effect Engineering for Aripiprazole (Abilify)")

with st.sidebar:
    st.header("📋 Clinical Inputs")
    dose_date = st.date_input("Dose Date", datetime(2025, 12, 9))
    dose_time = st.time_input("Dose Time", value=datetime.strptime("10:00 PM", "%I:%M %p").time())
    selected_se = st.selectbox("Observed Side Effect", list(ABILIFY_DATA["side_effects"].keys()))
    counter_med = st.selectbox("Select Counter Medication", list(COUNTER_MEDS.keys()))

# --- 4. DATA PROCESSING ---
dt_dose = datetime.combine(dose_date, dose_time)
t_plot = np.linspace(0, 48, 1000)

abilify_conc = pk_model(t_plot, ABILIFY_DATA["dose"], ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])
se_info = ABILIFY_DATA["side_effects"][selected_se]
se_curve = pk_model(t_plot - se_info["lag"], 80, ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

# Smart Recommender Logic: Predictive Onset
se_onset_hour = t_plot[np.where(se_curve > (np.max(se_curve) * 0.15))[0][0]] if any(se_curve > 0) else 0
predicted_peak_time = dt_dose + timedelta(hours=t_plot[np.argmax(se_curve)])

counter_conc = np.zeros_like(t_plot)
rec_time_str = "None Selected"

if counter_med != "None":
    c_data = COUNTER_MEDS[counter_med]
    # Recommendation: Counter-med peak should occur just as SE starts rising
    optimal_offset = se_onset_hour 
    counter_conc = pk_model(t_plot - optimal_offset, 110, c_data["ka"], c_data["ke"])
    rec_time = dt_dose + timedelta(hours=optimal_offset)
    rec_time_str = rec_time.strftime("%m/%d %I:%M %p")

# --- 5. DASHBOARD METRICS ---
m1, m2, m3 = st.columns(3)
with m1: st.metric("Side Effect Peak", predicted_peak_time.strftime("%I:%M %p"))
with m2: st.metric("Shield Onset Needed By", (dt_dose + timedelta(hours=se_onset_hour)).strftime("%I:%M %p"))
with m3: st.metric("Recommended Dose Time", rec_time_str)

# --- 6. MODERN VISUALIZATION ---
fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0e1117')
ax.set_facecolor('#0e1117')

# Axis formatting
ax.tick_params(colors='white', which='both')
for spine in ax.spines.values(): spine.set_color('#333333')

# Plotting
ax.plot(t_plot, abilify_conc, label="Abilify Conc.", color="#3498db", alpha=0.4, lw=1)
ax.plot(t_plot, se_curve, label=f"{selected_se} (Full Potential Risk)", color=se_info["color"], lw=2, alpha=0.8)

if counter_med != "None":
    ax.plot(t_plot, counter_conc, label=f"Counter: {counter_med}", color="#2ecc71", lw=2)
    
    # Mitigation logic
    mitigated = np.minimum(se_curve, counter_conc)
    unmitigated = se_curve - mitigated
    
    # Shading
    ax.fill_between(t_plot, 0, mitigated, color="#f1c40f", alpha=0.4, label="Mitigated Area")
    ax.fill_between(t_plot, mitigated, se_curve, color="#e74c3c", alpha=0.2, label="Unmitigated Risk")

# Clean X-Axis Time Labels
tick_indices = np.arange(0, 49, 6)
time_labels = [(dt_dose + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in tick_indices]
ax.set_xticks(tick_indices)
ax.set_xticklabels(time_labels, color='white', rotation=0)

ax.set_ylabel("Clinical Intensity", color='white')
ax.legend(facecolor='#1f2937', edgecolor='#333333', labelcolor='white', loc='upper right', fontsize='small')
plt.grid(color='#333333', linestyle='--', alpha=0.3)

st.pyplot(fig)
