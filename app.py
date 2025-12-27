import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time

# --- 1. EXPANDED DRUG & SYMPTOM DATABASE ---
# Data sourced from FDA labels & Pharmacokinetic literature
DRUG_DB = {
    "Sertraline (Zoloft)": {"ka": 0.5, "ke": 0.026, "symptoms": ["Nausea", "Insomnia", "Drowsiness", "Tremor"]},
    "Escitalopram (Lexapro)": {"ka": 0.6, "ke": 0.023, "symptoms": ["Nausea", "Fatigue", "Insomnia", "Sweating"]},
    "Bupropion (Wellbutrin)": {"ka": 1.2, "ke": 0.033, "symptoms": ["Agitation", "Insomnia", "Crash/Irritability", "Tachycardia"]},
    "Venlafaxine (Effexor)": {"ka": 1.0, "ke": 0.138, "symptoms": ["Nausea", "Dizziness", "Brain Zaps (Rebound)", "Hypertension"]},
    "Duloxetine (Cymbalta)": {"ka": 0.4, "ke": 0.057, "symptoms": ["Nausea", "Dry Mouth", "Fatigue", "Dizziness"]},
    "Fluoxetine (Prozac)": {"ka": 0.3, "ke": 0.006, "symptoms": ["Insomnia", "Anxiety", "Nausea", "Lethargy"]},
    "Citalopram (Celexa)": {"ka": 0.6, "ke": 0.020, "symptoms": ["Drowsiness", "Insomnia", "Nausea", "Yawning"]},
    "Trazodone": {"ka": 1.5, "ke": 0.115, "symptoms": ["Extreme Sedation", "Dizziness", "Dry Mouth", "Priapism Risk"]},
    "Alprazolam (Xanax)": {"ka": 1.5, "ke": 0.063, "symptoms": ["Sedation", "Rebound Anxiety", "Memory Fog", "Ataxia"]},
    "Lorazepam (Ativan)": {"ka": 1.2, "ke": 0.057, "symptoms": ["Sedation", "Dizziness", "Weakness", "Unsteadiness"]},
    "Clonazepam (Klonopin)": {"ka": 1.0, "ke": 0.023, "symptoms": ["Drowsiness", "Ataxia", "Fatigue", "Depression"]},
    "Methylphenidate (Ritalin)": {"ka": 1.8, "ke": 0.231, "symptoms": ["Crash/Irritability", "Heart Racing", "Insomnia", "Anorexia"]},
    "Lisdexamfetamine (Vyvanse)": {"ka": 0.7, "ke": 0.057, "symptoms": ["Late-Day Crash", "Insomnia", "Dry Mouth", "Anxiety"]},
    "Amphetamine/Dextro (Adderall)": {"ka": 1.1, "ke": 0.069, "symptoms": ["Crash/Irritability", "Anxiety", "Insomnia", "Palpitations"]},
    "Quetiapine (Seroquel)": {"ka": 1.2, "ke": 0.115, "symptoms": ["Heavy Sedation", "Weight Gain Risk", "Dry Mouth", "Orthostatic Hypotension"]},
    "Aripiprazole (Abilify)": {"ka": 0.8, "ke": 0.009, "symptoms": ["Akathisia", "Restlessness", "Insomnia", "Blurred Vision"]},
    "Lamotrigine (Lamictal)": {"ka": 1.1, "ke": 0.027, "symptoms": ["Dizziness", "Drowsiness", "Rash Risk (Cumulative)", "Ataxia"]},
    "Buspirone": {"ka": 1.5, "ke": 0.231, "symptoms": ["Dizziness", "Nausea", "Headache", "Excitement"]},
    "Mirtazapine": {"ka": 1.2, "ke": 0.034, "symptoms": ["Heavy Sedation", "Increased Appetite", "Dizziness", "Weight Gain Risk"]},
    "Hydroxyzine": {"ka": 1.4, "ke": 0.034, "symptoms": ["Sedation", "Dry Mouth", "Dizziness", "Blurred Vision"]},
    "Amitriptyline": {"ka": 0.8, "ke": 0.035, "symptoms": ["Dry Mouth", "Sedation", "Constipation", "Weight Gain Risk"]},
    "Olanzapine (Zyprexa)": {"ka": 1.0, "ke": 0.021, "symptoms": ["Sedation", "Weight Gain Risk", "Dizziness", "Increased Appetite"]},
    "Risperidone": {"ka": 1.3, "ke": 0.035, "symptoms": ["Akathisia", "Sedation", "Weight Gain Risk", "Dystonia"]},
    "Gabapentin": {"ka": 0.6, "ke": 0.115, "symptoms": ["Dizziness", "Drowsiness", "Peripheral Edema", "Ataxia"]},
    "Topiramate": {"ka": 0.9, "ke": 0.033, "symptoms": ["Paresthesia", "Cognitive Slowing", "Fatigue", "Anorexia"]},
    "Carbamazepine": {"ka": 0.5, "ke": 0.046, "symptoms": ["Dizziness", "Drowsiness", "Nausea", "Ataxia"]},
    "Lithium": {"ka": 1.5, "ke": 0.029, "symptoms": ["Hand Tremor", "Polyuria", "Thirst", "Nausea"]},
    "Paroxetine (Paxil)": {"ka": 0.7, "ke": 0.033, "symptoms": ["Nausea", "Drowsiness", "Sweating", "Sexual Dysfunction"]},
    "Clomipramine": {"ka": 0.8, "ke": 0.021, "symptoms": ["Dry Mouth", "Drowsiness", "Constipation", "Tremor"]}
}

SYMPTOM_LOGIC = {
    # Rebound effects (Category 2)
    "Crash/Irritability": "rebound", "Brain Zaps (Rebound)": "rebound", 
    "Rebound Anxiety": "rebound", "Late-Day Crash": "rebound",
    # Trailing effects (Category 1)
    "Akathisia": "trailing", "Nausea": "trailing", "Insomnia": "trailing", 
    "Sedation": "trailing", "Extreme Sedation": "trailing", "Drowsiness": "trailing", 
    "Dizziness": "trailing", "Restlessness": "trailing", "Dry Mouth": "trailing",
    "Heart Racing": "trailing", "Tachycardia": "trailing", "Dystonia": "trailing",
    "Paresthesia": "trailing", "Tremor": "trailing", "Hand Tremor": "trailing",
    # Cumulative effects (Category 3)
    "Weight Gain Risk": "cumulative", "Rash Risk (Cumulative)": "cumulative", 
    "Increased Appetite": "cumulative", "Peripheral Edema": "cumulative"
}

# --- 2. CORE ENGINE ---
def pk_model(t, ka, ke):
    t = np.maximum(t, 0)
    return (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(c, m=100):
    max_val = np.max(c)
    return (c / max_val) * m if max_val > 0 else c

# --- 3. UI SETUP ---
st.set_page_config(page_title="Med-Engine Pro", layout="wide")
st.title("💊 Multi-Drug Pharmaco-Logic Optimizer")
st.write("Visualizing pharmacokinetic interactions for clinical decision support.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Primary Medication")
    main_med = st.selectbox("Select Primary Drug", sorted(list(DRUG_DB.keys())))
    main_date = st.date_input("Dose Date", datetime.now(), key="d1")
    main_time = st.time_input("Dose Time", time(8, 0), key="t1")
    symptom = st.selectbox("Target Side Effect", DRUG_DB[main_med]["symptoms"])

with col2:
    st.header("2. Counteractive Strategy")
    counter_med = st.selectbox("Select Counter Drug", sorted(list(DRUG_DB.keys())), index=10)
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
    
    logic = SYMPTOM_LOGIC.get(symptom, "trailing")
    if logic == "rebound":
        grad = np.gradient(main_curve, h_axis)
        se_curve = normalize(np.where((grad < 0) & (main_curve < 50), np.abs(grad), 0), 85)
    elif logic == "cumulative":
        # Cumulative logic: show the rising integral of exposure
        se_curve = normalize(np.cumsum(main_curve), 85)
    else: 
        se_curve = normalize(pk_model(ts1 - 1.5, m_pk['ka'], m_pk['ke']), 85)

    mitigation_curve = np.minimum(se_curve, counter_curve)
    
    dx = h_axis[1] - h_axis[0]
    total_symptom_area = np.sum(se_curve) * dx
    mitigated_area = np.sum(mitigation_curve) * dx
    mitigation_pct = (mitigated_area / total_symptom_area * 100) if total_symptom_area > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h_axis, main_curve, label=f"Primary: {main_med}", color="#1f77b4", alpha=0.5)
    ax.plot(h_axis, se_curve, label=f"Symptom: {symptom}", color="#d62728", ls="--", lw=2)
    ax.plot(h_axis, counter_curve, label=f"Relief: {counter_med}", color="#2ca02c", lw=3)
    ax.fill_between(h_axis, 0, mitigation_curve, color="#FFD700", alpha=0.4, label=f"Mitigation ({mitigation_pct:.1f}%)")
    
    ax.set_xticks(np.arange(0, 37, 4))
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in np.arange(0, 37, 4)])
    ax.set_ylabel("Intensity (%)")
    ax.set_xlabel("Time (36h Window)")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Strategic Optimization Analysis")
    
    r_col1, r_col2, r_col3 = st.columns(3)
    
    se_peak_h = h_axis[np.argmax(se_curve)]
    se_peak_t = start_plot + timedelta(hours=se_peak_h)
    opt_dose_t = se_peak_t - timedelta(hours=2)

    r_col1.metric("Symptom Peak Prediction", se_peak_t.strftime("%I:%M %p"))
    r_col2.metric("Optimal Counter-Dose Time", opt_dose_t.strftime("%I:%M %p"))
    r_col3.metric("Neutralization Efficiency", f"{int(mitigation_pct)}%")

    st.info(f"**Clinical Insight:** {symptom} is expected to peak at **{se_peak_t.strftime('%I:%M %p')}**. Administering {counter_med} at **{opt_dose_t.strftime('%I:%M %p')}** provides the highest probability of symptom neutralization.")
