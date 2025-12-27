import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time

# --- 1. DRUG & SYMPTOM DATABASE ---
DRUG_DB = {
    "Sertraline (Zoloft)": {"ka": 0.5, "ke": 0.026, "symptoms": ["Nausea", "Insomnia", "Drowsiness"]},
    "Escitalopram (Lexapro)": {"ka": 0.6, "ke": 0.023, "symptoms": ["Nausea", "Fatigue", "Insomnia"]},
    "Bupropion (Wellbutrin)": {"ka": 1.2, "ke": 0.033, "symptoms": ["Agitation", "Insomnia", "Crash/Irritability"]},
    "Venlafaxine (Effexor)": {"ka": 1.0, "ke": 0.138, "symptoms": ["Nausea", "Dizziness", "Brain Zaps (Rebound)"]},
    "Duloxetine (Cymbalta)": {"ka": 0.4, "ke": 0.057, "symptoms": ["Nausea", "Dry Mouth", "Fatigue"]},
    "Fluoxetine (Prozac)": {"ka": 0.3, "ke": 0.006, "symptoms": ["Insomnia", "Anxiety", "Nausea"]},
    "Citalopram (Celexa)": {"ka": 0.6, "ke": 0.020, "symptoms": ["Drowsiness", "Insomnia", "Nausea"]},
    "Trazodone": {"ka": 1.5, "ke": 0.115, "symptoms": ["Extreme Sedation", "Dizziness", "Dry Mouth"]},
    "Alprazolam (Xanax)": {"ka": 1.5, "ke": 0.063, "symptoms": ["Sedation", "Rebound Anxiety", "Memory Fog"]},
    "Lorazepam (Ativan)": {"ka": 1.2, "ke": 0.057, "symptoms": ["Sedation", "Dizziness", "Weakness"]},
    "Clonazepam (Klonopin)": {"ka": 1.0, "ke": 0.023, "symptoms": ["Drowsiness", "Ataxia", "Fatigue"]},
    "Methylphenidate (Ritalin)": {"ka": 1.8, "ke": 0.231, "symptoms": ["Crash/Irritability", "Heart Racing", "Insomnia"]},
    "Lisdexamfetamine (Vyvanse)": {"ka": 0.7, "ke": 0.057, "symptoms": ["Late-Day Crash", "Insomnia", "Decreased Appetite"]},
    "Amphetamine/Dextro (Adderall)": {"ka": 1.1, "ke": 0.069, "symptoms": ["Crash/Irritability", "Anxiety", "Insomnia"]},
    "Quetiapine (Seroquel)": {"ka": 1.2, "ke": 0.115, "symptoms": ["Heavy Sedation", "Weight Gain Risk", "Dry Mouth"]},
    "Aripiprazole (Abilify)": {"ka": 0.8, "ke": 0.009, "symptoms": ["Akathisia", "Restlessness", "Insomnia"]},
    "Lamotrigine (Lamictal)": {"ka": 1.1, "ke": 0.027, "symptoms": ["Dizziness", "Drowsiness", "Rash Risk (Cumulative)"]},
    "Buspirone": {"ka": 1.5, "ke": 0.231, "symptoms": ["Dizziness", "Nausea", "Headache"]},
    "Mirtazapine": {"ka": 1.2, "ke": 0.034, "symptoms": ["Heavy Sedation", "Increased Appetite", "Dizziness"]},
    "Hydroxyzine": {"ka": 1.4, "ke": 0.034, "symptoms": ["Sedation", "Dry Mouth", "Dizziness"]}
}

SYMPTOM_LOGIC = {
    "Crash/Irritability": "rebound", "Brain Zaps (Rebound)": "rebound", "Rebound Anxiety": "rebound", "Late-Day Crash": "rebound",
    "Akathisia": "trailing", "Nausea": "trailing", "Insomnia": "trailing", "Sedation": "trailing", 
    "Extreme Sedation": "trailing", "Drowsiness": "trailing", "Dizziness": "trailing", "Restlessness": "trailing",
    "Weight Gain Risk": "cumulative", "Rash Risk (Cumulative)": "cumulative"
}

# --- 2. CORE ENGINE ---
def pk_model(t, ka, ke):
    t = np.maximum(t, 0)
    return (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(c, m=100):
    return (c / np.max(c)) * m if np.max(c) > 0 else c

# --- 3. UI SETUP ---
st.set_page_config(page_title="Global Pharmaco-Logic SaaS", layout="wide")
st.title("🧠 Clinical Precision Medication Manager")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Primary Prescription")
    main_med = st.selectbox("Select Primary Drug", list(DRUG_DB.keys()))
    main_date = st.date_input("Dose Date", datetime.now(), key="d1")
    main_time = st.time_input("Dose Time", time(8, 0), key="t1")
    symptom = st.selectbox("Target Side Effect to Mitigate", DRUG_DB[main_med]["symptoms"])

with col2:
    st.header("2. Counteractive Strategy")
    counter_med = st.selectbox("Select Counter Drug", list(DRUG_DB.keys()), index=10)
    counter_date = st.date_input("Counter Dose Date", datetime.now(), key="d2")
    counter_time = st.time_input("Counter Dose Time", time(14, 0), key="t2")

dt1 = datetime.combine(main_date, main_time)
dt2 = datetime.combine(counter_date, counter_time)

# --- 4. CALCULATION & GRAPHING ---
if dt1 and dt2:
    start_plot = min(dt1, dt2) - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000)
    ts1 = np.array([((start_plot + timedelta(hours=h)) - dt1).total_seconds()/3600 for h in h_axis])
    ts2 = np.array([((start_plot + timedelta(hours=h)) - dt2).total_seconds()/3600 for h in h_axis])

    m_pk = DRUG_DB[main_med]
    c_pk = DRUG_DB[counter_med]
    
    main_curve = normalize(pk_model(ts1, m_pk['ka'], m_pk['ke']))
    counter_curve = normalize(pk_model(ts2, c_pk['ka'], c_pk['ke']))
    
    # Symptom Curve Logic
    logic = SYMPTOM_LOGIC.get(symptom, "trailing")
    if logic == "rebound":
        grad = np.gradient(main_curve, h_axis)
        se_curve = normalize(np.where((grad < 0) & (main_curve < 50), np.abs(grad), 0), 85)
    elif logic == "cumulative":
        se_curve = normalize(np.cumsum(main_curve), 85)
    else: # trailing
        se_curve = normalize(pk_model(ts1 - 1.5, m_pk['ka'], m_pk['ke']), 85)

    # Mitigation calculation (The "Secret Sauce")
    mitigation_curve = np.minimum(se_curve, counter_curve)
    
    # Calculate Areas for Percentage
    total_symptom_area = np.trapz(se_curve, h_axis)
    mitigated_area = np.trapz(mitigation_curve, h_axis)
    
    mitigation_pct = 0
    if total_symptom_area > 0:
        mitigation_pct = (mitigated_area / total_symptom_area) * 100

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h_axis, main_curve, label=f"Primary: {main_med}", color="#1f77b4", alpha=0.7)
    ax.plot(h_axis, se_curve, label=f"Symptom: {symptom}", color="#d62728", ls="--")
    ax.plot(h_axis, counter_curve, label=f"Relief: {counter_med}", color="#2ca02c", lw=3)
    ax.fill_between(h_axis, 0, mitigation_curve, color="#FFD700", alpha=0.4, label=f"Mitigation ({mitigation_pct:.1f}%)")
    
    # Visual Polish
    ax.set_xticks(np.arange(0, 37, 4))
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in np.arange(0, 37, 4)])
    ax.set_ylabel("Intensity (%)")
    ax.set_xlabel("Time (36h Window)")
    ax.legend(loc='upper right')
    st.pyplot(fig)

    # --- 5. SMART RECOMMENDER & ANALYTICS ---
    st.markdown("---")
    st.subheader("📊 Strategic Analysis & Optimization")
    
    r_col1, r_col2, r_col3 = st.columns(3)
    
    # Finding peak and optimal time
    se_peak_h = h_axis[np.argmax(se_curve)]
    se_peak_t = start_plot + timedelta(hours=se_peak_h)
    opt_dose_t = se_peak_t - timedelta(hours=2) # Lead time logic

    r_col1.metric("Side Effect Peak", se_peak_t.strftime("%I:%M %p"))
    r_col2.metric("Optimal Strategy Time", opt_dose_t.strftime("%I:%M %p"))
    r_col3.metric("Mitigation Score", f"{int(mitigation_pct)}%")

    st.info(f"**Clinical Insight:** Based on current inputs, {int(mitigation_pct)}% of the predicted {symptom} intensity is covered. To reach 100% neutralization, consider adjusting the {counter_med} dose to **{opt_dose_t.strftime('%I:%M %p')}**.")
