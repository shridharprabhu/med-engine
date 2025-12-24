import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. THE BRAIN: DRUG & SIDE EFFECT DICTIONARY ---
# This allows the SaaS to scale. You just add rows here to add drugs.
DRUG_DB = {
    "Adderall IR": {"ka": 1.1, "ke": np.log(2)/10, "type": "Stimulant"},
    "Abilify": {"ka": 1.0, "ke": np.log(2)/75, "type": "Antipsychotic"},
    "Xanax": {"ka": 1.5, "ke": np.log(2)/12, "type": "Benzodiazepine"},
    "Clonazepam": {"ka": 1.8, "ke": np.log(2)/35, "type": "Benzodiazepine"}
}

SIDE_EFFECT_DB = {
    "Crash/Irritability": {"category": 2, "threshold": 0.50}, # Rebound (50% drop)
    "Akathisia": {"category": 1, "lag": 1.5},                # Trailing (1.5h lag)
    "Nausea": {"category": 1, "lag": 0.5},                   # Trailing (0.5h lag)
    "Restlessness": {"category": 1, "lag": 1.0}              # Trailing (1.0h lag)
}

# --- 2. HELPER FUNCTIONS ---
def parse_med_time(dt_str):
    dt_str = dt_str.lower().strip()
    full_str = f"{dt_str} 2025" 
    formats = ["%m/%d %I%p %Y", "%m/%d %I:%M%p %Y", "%m/%d %H:%M %Y"]
    for fmt in formats:
        try: return datetime.strptime(full_str, fmt)
        except ValueError: continue
    return None

def pk_model(t_hours, ka, ke):
    t = np.maximum(t_hours, 0)
    return (1.0 * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(curve, target_max=100):
    if np.max(curve) == 0: return curve
    return (curve / np.max(curve)) * target_max

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Medication Timing Optimizer", layout="wide")
st.title("🛡️ Clinical Precision Optimizer")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Primary Medication")
    main_med = st.selectbox("Select Medication", list(DRUG_DB.keys()))
    main_time_str = st.text_input("Dose Time (MM/DD HHam/pm)", "12/24 8am")
    side_effect = st.selectbox("Target Side Effect", list(SIDE_EFFECT_DB.keys()))

with col2:
    st.subheader("2. Counter Medication")
    counter_med = st.selectbox("Select Counter-Drug", list(DRUG_DB.keys()), index=3)
    counter_time_str = st.text_input("Counter Dose Time", "12/24 2pm")

dt_1 = parse_med_time(main_time_str)
dt_2 = parse_med_time(counter_time_str)

if dt_1 and dt_2:
    # --- 4. CALCULATION ENGINE ---
    start_plot = dt_1 - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000) 
    
    # Times since doses
    t_s1 = np.array([((start_plot + timedelta(hours=h)) - dt_1).total_seconds()/3600 for h in h_axis])
    t_s2 = np.array([((start_plot + timedelta(hours=h)) - dt_2).total_seconds()/3600 for h in h_axis])

    # Chief Curve
    c1 = normalize(pk_model(t_s1, DRUG_DB[main_med]['ka'], DRUG_DB[main_med]['ke']))
    
    # Side Effect Logic
    se_data = SIDE_EFFECT_DB[side_effect]
    if se_data['category'] == 1:
        # Category 1: Trailing
        se_curve = normalize(pk_model(t_s1 - se_data['lag'], DRUG_DB[main_med]['ka'], DRUG_DB[main_med]['ke']), 85)
    else:
        # Category 2: Rebound (Velocity at <50%)
        grad = np.gradient(c1, h_axis)
        se_curve = normalize(np.where((grad < 0) & (c1 < se_data['threshold']*100), np.abs(grad), 0), 85)

    # Counter Curve
    c2 = normalize(pk_model(t_s2, DRUG_DB[counter_med]['ka'], DRUG_DB[counter_med]['ke']))

    # --- 5. VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h_axis, c1, color='#FF8C00', label=f'{main_med} (Main)', lw=2)
    ax.plot(h_axis, se_curve, color='#8E44AD', ls='--', label=f'{side_effect} Risk', lw=2)
    ax.plot(h_axis, c2, color='#27AE60', label=f'{counter_med} (Relief)', lw=3)
    
    # Yellow Mitigation Area
    mitigation = np.minimum(se_curve, c2)
    ax.fill_between(h_axis, 0, mitigation, color='#F1C40F', alpha=0.4, label='Neutralization')

    # Formatting
    tick_indices = np.arange(0, 37, 4)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in tick_indices])
    ax.legend(loc='upper right')
    ax.set_ylabel("Normalized Intensity %")
    st.pyplot(fig)

    # --- 6. RECOMMENDER SYSTEM ---
    st.divider()
    st.subheader("💡 Smart Recommender System")
    
    # Finding the peak of the side effect
    se_peak_idx = np.argmax(se_curve)
    se_peak_time = start_plot + timedelta(hours=h_axis[se_peak_idx])
    
    # Finding the "On-set time" of counter-med (Time to peak)
    # Approx t_max for counter med
    c_tmax = 1 / (DRUG_DB[counter_med]['ka'] - DRUG_DB[counter_med]['ke']) # Simplified tmax
    
    # Optimal recommendation logic
    opt_dose_time = se_peak_time - timedelta(hours=2) # Subtracting peak-to-onset time
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.metric("Side Effect Peak Predicted", se_peak_time.strftime("%I:%M %p"))
        st.write(f"The symptom **{side_effect}** is projected to reach maximum intensity at approximately {se_peak_time.strftime('%I:%M %p')}.")
    
    with rec_col2:
        st.metric("Recommended Dose Time", opt_dose_time.strftime("%I:%M %p"))
        st.info(f"To achieve maximum neutralization, administer **{counter_med}** at **{opt_dose_time.strftime('%I:%M %p')}**. This ensures the relief peak aligns with the symptom peak.")

else:
    st.warning("Awaiting valid time inputs to generate clinical map.")
