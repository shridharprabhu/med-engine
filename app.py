import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. PK ENGINE ---
def pk_model(t, dose, ka, ke):
    t = np.maximum(t, 0)
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

# --- 2. DATA LIBRARY (The "Moat") ---
# Tmax and Half-life sourced from FDA Structured Product Labeling (SPL)
MEDS = {
    "Abilify": {"ka": 1.0, "ke": np.log(2)/75, "tmax": 4.0},
    "Seroquel": {"ka": 1.5, "ke": np.log(2)/7, "tmax": 1.5},
    "Adderall IR": {"ka": 2.0, "ke": np.log(2)/10, "tmax": 3.0},
    "Zoloft": {"ka": 0.5, "ke": np.log(2)/26, "tmax": 8.0},
    "Lexapro": {"ka": 0.8, "ke": np.log(2)/30, "tmax": 5.0},
    "Vyvanse": {"ka": 0.9, "ke": np.log(2)/12, "tmax": 3.5},
    "Prozac": {"ka": 0.4, "ke": np.log(2)/120, "tmax": 7.0},
    "Wellbutrin": {"ka": 1.2, "ke": np.log(2)/21, "tmax": 1.5},
    "Risperdal": {"ka": 2.5, "ke": np.log(2)/20, "tmax": 1.0},
    "Zyprexa": {"ka": 0.7, "ke": np.log(2)/33, "tmax": 6.0},
    "Haldol": {"ka": 1.0, "ke": np.log(2)/24, "tmax": 4.0},
    "Effexor": {"ka": 1.3, "ke": np.log(2)/5, "tmax": 2.0},
    "Latuda": {"ka": 1.1, "ke": np.log(2)/18, "tmax": 2.0},
    "Concerta": {"ka": 0.3, "ke": np.log(2)/4, "tmax": 8.0},
    "Ritalin": {"ka": 2.0, "ke": np.log(2)/3, "tmax": 2.0},
    "Paxil": {"ka": 0.6, "ke": np.log(2)/21, "tmax": 5.0},
    "Lithium": {"ka": 1.5, "ke": np.log(2)/24, "tmax": 2.0},
    "Xanax": {"ka": 3.0, "ke": np.log(2)/11, "tmax": 1.5},
    "Ativan": {"ka": 2.0, "ke": np.log(2)/12, "tmax": 2.0},
    "Klonopin": {"ka": 1.8, "ke": np.log(2)/35, "tmax": 2.5}
}

SIDE_EFFECTS = {
    "Akathisia": {"lag": 2.0, "color": "#e74c3c"},
    "Nausea": {"lag": 0.5, "color": "#9b59b6"},
    "Insomnia": {"lag": 4.5, "color": "#e67e22"},
    "Dizziness": {"lag": 1.0, "color": "#a04000"},
    "Fatigue": {"lag": 5.0, "color": "#8e44ad"},
    "Anxiety": {"lag": 1.5, "color": "#f06292"},
    "Tremor": {"lag": 2.0, "color": "#c0392b"},
    "Headache": {"lag": 2.5, "color": "#273746"},
    "Brain Zaps": {"lag": 6.0, "color": "#1abc9c"},
    "The Crash (Rebound)": {"lag": 7.0, "color": "#7f8c8d"}
}

COUNTERS = {
    "None": None,
    "Clonazepam": {"ka": 1.8, "ke": np.log(2)/35, "tmax": 2.5},
    "Propranolol": {"ka": 2.5, "ke": np.log(2)/4, "tmax": 1.5},
    "Melatonin": {"ka": 3.0, "ke": np.log(2)/1, "tmax": 0.5},
    "Guanfacine": {"ka": 1.2, "ke": np.log(2)/17, "tmax": 3.0},
    "Zofran": {"ka": 2.8, "ke": np.log(2)/3.5, "tmax": 1.5},
    "Benadryl": {"ka": 2.2, "ke": np.log(2)/8, "tmax": 2.0},
    "Hydroxyzine": {"ka": 2.0, "ke": np.log(2)/20, "tmax": 2.0},
    "Trazodone": {"ka": 1.5, "ke": np.log(2)/10, "tmax": 2.0},
    "Magnesium": {"ka": 0.8, "ke": np.log(2)/12, "tmax": 4.0},
    "Xanax": {"ka": 3.0, "ke": np.log(2)/11, "tmax": 1.0}
}

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Med-Engine v3", layout="wide")
st.markdown("<style>.stApp { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("💊 Primary Medication")
    p_med = st.selectbox("Select Drug", list(MEDS.keys()))
    p_date = st.date_input("Dose Date (Primary)", datetime(2025, 12, 9))
    p_time = st.time_input("Dose Time (Primary)", value=datetime.strptime("10:00 PM", "%I:%M %p").time())
    
    st.header("🚩 Side Effect")
    se_name = st.selectbox("Observed Symptom", list(SIDE_EFFECTS.keys()))
    
    st.header("🛡️ Counter Medication")
    c_med = st.selectbox("Select Relief Drug", list(COUNTERS.keys()))
    c_date = st.date_input("Dose Date (Counter)", datetime(2025, 12, 9))
    c_time = st.time_input("Dose Time (Counter)", value=datetime.strptime("11:30 PM", "%I:%M %p").time())

# --- 4. ENGINE CALCULATIONS ---
t_plot = np.linspace(0, 48, 1000)
dt_p = datetime.combine(p_date, p_time)
dt_c = datetime.combine(c_date, c_time)

# Primary & SE Curves
p_pk = MEDS[p_med]
se_pk = SIDE_EFFECTS[se_name]
p_conc = pk_model(t_plot, 100, p_pk["ka"], p_pk["ke"])
se_conc = pk_model(t_plot - se_pk["lag"], 85, p_pk["ka"], p_pk["ke"])

# Counter Curve
c_conc = np.zeros_like(t_plot)
offset_c = (dt_c - dt_p).total_seconds() / 3600
if c_med != "None":
    c_pk = COUNTERS[c_med]
    c_conc = pk_model(t_plot - offset_c, 110, c_pk["ka"], c_pk["ke"])

# Smart Recommender
se_onset_h = t_plot[np.where(se_conc > (np.max(se_conc)*0.15))[0][0]] if any(se_conc > 0) else 0
rec_time = dt_p + timedelta(hours=se_onset_h - (COUNTERS[c_med]["tmax"] if c_med != "None" else 0))

# --- 5. DASHBOARD ---
c1, c2 = st.columns(2)
c1.metric("Predicted Symptom Peak", (dt_p + timedelta(hours=t_plot[np.argmax(se_conc)])).strftime("%I:%M %p"))
c2.metric("Optimal Counter-Dose Time", rec_time.strftime("%I:%M %p") if c_med != "None" else "N/A")

fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0e1117')
ax.set_facecolor('#0e1117')
ax.plot(t_plot, p_conc, color="#3498db", alpha=0.3, label=f"{p_med} Blood Level")
ax.plot(t_plot, se_conc, color=se_pk["color"], lw=2, label=f"Risk: {se_name}")

if c_med != "None":
    ax.plot(t_plot, c_conc, color="#2ecc71", lw=2, label=f"Shield: {c_med}")
    mitigated = np.minimum(se_conc, c_conc)
    ax.fill_between(t_plot, 0, mitigated, color="#f1c40f", alpha=0.4, label="Mitigated")
    ax.fill_between(t_plot, mitigated, se_conc, color="#e74c3c", alpha=0.15, label="Exposed Risk")

# X-Axis Styling
ticks = np.arange(0, 49, 6)
ax.set_xticks(ticks)
ax.set_xticklabels([(dt_p + timedelta(hours=int(h))).strftime("%m/%d\n%I%p") for h in ticks], color='white')
ax.legend(facecolor='#1f2937', labelcolor='white')
st.pyplot(fig)
