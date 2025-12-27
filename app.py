import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. DRUG INTELLIGENCE DATABASE ---
# Ka: Absorption rate, Ke: Elimination rate (based on Half-life)
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

# Categorizing Symptom Behavior
SYMPTOM_LOGIC = {
    "Crash/Irritability": "rebound", "Brain Zaps (Rebound)": "rebound", "Rebound Anxiety": "rebound", "Late-Day Crash": "rebound",
    "Akathisia": "trailing", "Nausea": "trailing", "Insomnia": "trailing", "Sedation": "trailing", 
    "Extreme Sedation": "trailing", "Drowsiness": "trailing", "Dizziness": "trailing", "Restlessness": "trailing",
    "Weight Gain Risk": "cumulative", "Rash Risk (Cumulative)": "cumulative"
}

# --- 2. ENGINE FUNCTIONS ---
def parse_dt(s):
    try: return datetime.strptime(f"{s} 2025", "%m/%d %I%p")
    except: 
        try: return datetime.strptime(f"{s} 2025", "%m/%d %I:%M%p")
        except: return None

def pk_model(t, ka, ke):
    t = np.maximum(t, 0)
    return (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(c, m=100):
    return (c / np.max(c)) * m if np.max(c) > 0 else c

# --- 3. UI ---
st.set_page_config(page_title="Global Pharmaco-Logic SaaS", layout="wide")
st.title("🧠 Clinical Decision Support System (CDSS)")
st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    main_med = st.selectbox("Primary Medication", list(DRUG_DB.keys()))
    main_time = st.text_input("Dose Time (MM/DD 10am)", "12/27 8am")
    symptom = st.selectbox("Target Side Effect", DRUG_DB[main_med]["symptoms"])

with c2:
    counter_med = st.selectbox("Counter Medication", list(DRUG_DB.keys()), index=10) # Default to Clonazepam
    counter_time = st.text_input("Counter Dose Time", "12/27 2pm")

dt1, dt2 = parse_dt(main_time), parse_dt(counter_time)

if dt1 and dt2:
    start_plot = dt1 - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000)
    ts1 = np.array([((start_plot + timedelta(hours=h)) - dt1).total_seconds()/3600 for h in h_axis])
    ts2 = np.array([((start_plot + timedelta(hours=h)) - dt2).total_seconds()/3600 for h in h_axis])

    # Curves
    m_pk = DRUG_DB[main_med]
    c_pk = DRUG_DB[counter_med]
    
    main_curve = normalize(pk_model(ts1, m_pk['ka'], m_pk['ke']))
    counter_curve = normalize(pk_model(ts2, c_pk['ka'], c_pk['ke']))
    
    # Symptom Mapping
    logic = SYMPTOM_LOGIC.get(symptom, "trailing")
    if logic == "rebound":
        grad = np.gradient(main_curve, h_axis)
        se_curve = normalize(np.where((grad < 0) & (main_curve < 50), np.abs(grad), 0), 85)
    elif logic == "cumulative":
        se_curve = normalize(np.cumsum(main_curve), 85)
    else: # trailing
        se_curve = normalize(pk_model(ts1 - 1.5, m_pk['ka'], m_pk['ke']), 85)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h_axis, main_curve, label="Primary Drug", color="#1f77b4", lw=2)
    ax.plot(h_axis, se_curve, label=f"Symptom: {symptom}", color="#d62728", ls="--", lw=2)
    ax.plot(h_axis, counter_curve, label="Counter Drug", color="#2ca02c", lw=3)
    
    mitigation = np.minimum(se_curve, counter_curve)
    ax.fill_between(h_axis, 0, mitigation, color="#FFD700", alpha=0.4, label="Neutralization")
    
    ax.set_xticks(np.arange(0, 37, 4))
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d %I%p") for h in np.arange(0, 37, 4)])
    ax.set_ylabel("Intensity (%)")
    ax.legend()
    st.pyplot(fig)

    # --- 4. RECOMMENDER ---
    st.subheader("💡 Strategic Recommendation")
    se_peak_h = h_axis[np.argmax(se_curve)]
    se_peak_t = start_plot + timedelta(hours=se_peak_h)
    
    # Heuristic for counter-drug onset (Tmax approx 1.5 - 3h)
    opt_t = se_peak_t - timedelta(hours=2)
    
    r1, r2 = st.columns(2)
    r1.metric("Predicted Symptom Peak", se_peak_t.strftime("%I:%M %p"))
    r2.metric("Optimal Dose Time", opt_t.strftime("%I:%M %p"))
    st.info(f"Analysis: To neutralize the peak of **{symptom}**, the patient should take **{counter_med}** at **{opt_t.strftime('%I:%M %p')}**.")
else:
    st.error("Format Error: Use MM/DD Time (e.g., 12/27 8am)")
