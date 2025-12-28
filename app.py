import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. THE BRAIN: DRUG & SIDE EFFECT DICTIONARY ---
DRUG_DB = {
    "Adderall IR": {"ka": 1.1, "ke": np.log(2)/10, "type": "Stimulant"},
    "Abilify": {"ka": 1.0, "ke": np.log(2)/75, "type": "Antipsychotic"},
    "Xanax": {"ka": 1.5, "ke": np.log(2)/12, "type": "Benzodiazepine"},
    "Clonazepam": {"ka": 1.8, "ke": np.log(2)/35, "type": "Benzodiazepine"}
}

SIDE_EFFECT_DB = {
    "Crash/Irritability": {"category": 2, "threshold": 0.50},
    "Akathisia": {"category": 1, "lag": 1.5},
    "Nausea": {"category": 1, "lag": 0.5},
    "Restlessness": {"category": 1, "lag": 1.0}
}

# --- 2. HELPER FUNCTIONS ---
def parse_med_time(dt_str):
    dt_str = dt_str.lower().strip()
    full_str = f"{dt_str} 2025"
    formats = ["%m/%d %I%p", "%m/%d %I:%M%p", "%m/%d %H:%M"]
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

# Custom CSS for Modern Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stSelectbox, .stTextInput { border-radius: 8px; }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🛡️ Clinical Precision Optimizer")
st.caption("Advanced Decision Support for Chronopharmacology")

with st.container():
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.subheader("💊 Primary Therapy")
        main_med = st.selectbox("Select Medication", list(DRUG_DB.keys()))
        main_time_str = st.text_input("Dose Start (MM/DD HHam/pm)", "12/28 8am")
        side_effect = st.selectbox("Target Side Effect Profile", list(SIDE_EFFECT_DB.keys()))

    with col2:
        st.subheader("🛡️ Mitigation Strategy")
        counter_med = st.selectbox("Select Counter-Drug", list(DRUG_DB.keys()), index=3)
        counter_time_str = st.text_input("Counter-Dose Start", "12/28 12pm")

dt_1 = parse_med_time(main_time_str)
dt_2 = parse_med_time(counter_time_str)

if dt_1 and dt_2:
    # --- 4. ENGINE ---
    start_plot = dt_1 - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000)
    t_s1 = np.array([((start_plot + timedelta(hours=h)) - dt_1).total_seconds()/3600 for h in h_axis])
    t_s2 = np.array([((start_plot + timedelta(hours=h)) - dt_2).total_seconds()/3600 for h in h_axis])

    # Calculate Chief Curve
    c1 = normalize(pk_model(t_s1, DRUG_DB[main_med]['ka'], DRUG_DB[main_med]['ke']))

    # Side Effect Logic
    se_data = SIDE_EFFECT_DB[side_effect]
    if se_data['category'] == 1:
        se_curve = normalize(pk_model(t_s1 - se_data['lag'], DRUG_DB[main_med]['ka'], DRUG_DB[main_med]['ke']), 85)
    else:
        # Category 2: Rebound
        grad = np.gradient(c1, h_axis)
        se_curve = normalize(np.where((grad < 0) & (c1 < se_data['threshold']*100), np.abs(grad), 0), 85)

    # Counter Curve
    c2 = normalize(pk_model(t_s2, DRUG_DB[counter_med]['ka'], DRUG_DB[counter_med]['ke']))

    # --- 5. VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    ax.set_facecolor('#ffffff')

    ax.plot(h_axis, c1, color='#d1d1d1', label=f'{main_med} (Blood Conc.)', lw=1.5, alpha=0.7)
    ax.plot(h_axis, se_curve, color='#333333', ls='-', label=f'{side_effect} Baseline', lw=1, alpha=0.5)
    ax.plot(h_axis, c2, color='#2E7D32', label=f'{counter_med} (Relief Peak)', lw=2.5)

    # Shading: Yellow for Protected, Light Red for Exposed
    mitigation_val = np.minimum(se_curve, c2)
    ax.fill_between(h_axis, 0, mitigation_val, color='#FFD600', alpha=0.5, label='Symptom Neutralized')
    ax.fill_between(h_axis, mitigation_val, se_curve, color='#FFCDD2', alpha=0.5, label='Symptom Exposed (Unmitigated)')

    # Formatting X-Axis (Horizontal Labels)
    tick_indices = np.arange(0, 37, 4)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d\n%I%p") for h in tick_indices], fontsize=9)

    ax.set_ylabel("Intensity (%)", fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.1)

    st.pyplot(fig)

    # --- 6. ADVANCED RECOMMENDER ---
    st.divider()
    st.subheader("💡 Predictive Recommendations")

    # Recommender Logic: Find when the Side Effect FIRST exceeds a 5% baseline (The Onset)
    onset_threshold = 5.0
    se_onset_idx = np.where(se_curve > onset_threshold)[0]

    if len(se_onset_idx) > 0:
        se_onset_time = start_plot + timedelta(hours=h_axis[se_onset_idx[0]])

        # Optimal Time = Onset of Symptom minus Time-to-Peak (Tmax) of counter drug
        ka_c, ke_c = DRUG_DB[counter_med]['ka'], DRUG_DB[counter_med]['ke']
        t_max_c = np.log(ka_c / ke_c) / (ka_c - ke_c)

        opt_dose_time = se_onset_time - timedelta(hours=t_max_c)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Symptom Onset Expected", se_onset_time.strftime("%I:%M %p"))
        with m2:
            st.metric("Optimal Counter-Dose", opt_dose_time.strftime("%I:%M %p"))
        with m3:
            neutralized_pct = (np.sum(mitigation_val) / np.sum(se_curve)) * 100 if np.sum(se_curve) > 0 else 0
            st.metric("Neutralization Efficiency", f"{int(neutralized_pct)}%")

        st.info(f"**Clinical Insight:** {side_effect} begins developing at **{se_onset_time.strftime('%I:%M %p')}**. To ensure the {counter_med} relief peak matches the onset of the symptom, administer the dose at **{opt_dose_time.strftime('%I:%M %p')}**.")
    else:
        st.info("Side effect profile is negligible for this dosage timing.")

else:
    st.error("Please provide valid date/time entries in the left panel.")
