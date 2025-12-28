import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. CORE ENGINE: PHARMACOKINETICS ---
def pk_model(t, dose, ka, ke):
    t = np.maximum(t, 0)
    # Standard 1-compartment PK model
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

# --- 2. DATASET ---
ABILIFY_DATA = {
    "ka": 1.0, "ke": np.log(2) / 75, "dose": 100,
    "side_effects": {
        "Akathisia (Restlessness)": {"lag": 2.0, "color": "#e74c3c"},
        "Insomnia": {"lag": 4.0, "color": "#e67e22"},
        "Nausea": {"lag": 0.5, "color": "#9b59b6"},
        "Dizziness": {"lag": 1.0, "color": "#a04000"},
        "Fatigue/Somnolence": {"lag": 5.0, "color": "#8e44ad"},
        "Blurred Vision": {"lag": 2.0, "color": "#7f8c8d"},
        "Anxiety": {"lag": 1.5, "color": "#f06292"},
        "Tremor": {"lag": 2.5, "color": "#c0392b"},
        "Dry Mouth": {"lag": 3.0, "color": "#1abc9c"},
        "Headache": {"lag": 2.0, "color": "#273746"}
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

# Fixed the CSS and the incorrect parameter name (unsafe_allow_html)
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #3498db; }
    div[data-testid="stMetric"] { 
        background-color: #1f2937; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #3498db; 
    }
    </style>
    """, 
    unsafe_allow_html=True
)

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
# Generate side effect curve with the specific lag
se_curve = pk_model(t_plot - se_info["lag"], 80, ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

# Smart Recommender Logic: Proactive Mitigation
# Find when side effect crosses 15% of its peak
se_threshold = np.max(se_curve) * 0.15
onset_indices = np.where(se_curve > se_threshold)[0]
se_onset_hour = t_plot[onset_indices[0]] if len(onset_indices) > 0 else 0

predicted_peak_time = dt_dose + timedelta(hours=t_plot[np.argmax(se_curve)])

counter_conc = np.zeros_like(t_plot)
rec_time_str = "None Selected"

if counter_med != "None":
    c_data = COUNTER_MEDS[counter_med]
    # Set counter-med peak to occur at the side effect onset for early protection
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

# Axis styling
ax.tick_params(colors='white', which='both', labelsize=10)
for spine in ax.spines.values(): 
    spine.set_color('#333333')

# Plotting the data
ax.plot(t_plot, abilify_conc, label="Abilify Conc.", color="#3498db", alpha=0.4, lw=1)
ax.plot(t_plot, se_curve, label=f"{selected_se} (Side Effect Risk)", color=se_info["color"], lw=2.5, alpha=0.9)

if counter_med != "None":
    ax.plot(t_plot, counter_conc, label=f"Counter: {counter_med}", color="#2ecc71", lw=2.5)
    
    # Mathematical Shading Logic
    mitigated = np.minimum(se_curve, counter_conc)
    
    # Shading zones
    ax.fill_between(t_plot, 0, mitigated, color="#f1c40f", alpha=0.4, label="Mitigated Area")
    ax.fill_between(t_plot, mitigated, se_curve, color="#e74c3c", alpha=0.2, label="Unmitigated Risk")

# Clean X-Axis Time Labels (Horizontal)
tick_indices = np.arange(0, 49, 6)
time_labels = [(dt_dose + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in tick_indices]
ax.set_xticks(tick_indices)
ax.set_xticklabels(time_labels, color='white')

ax.set_ylabel("Clinical Intensity", color='white', fontsize=12)
ax.legend(facecolor='#1f2937', edgecolor='#333333', labelcolor='white', loc='upper right')
plt.grid(color='#333333', linestyle='--', alpha=0.3)

# Display in Streamlit
st.pyplot(fig)
