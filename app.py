import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. CORE ENGINE: PHARMACOKINETICS ---
def pk_model(t, dose, ka, ke):
    """Standard 1-Compartment PK Model"""
    t = np.maximum(t, 0)
    # C(t) = [D*ka / (Vd*(ka-ke))] * (exp(-ke*t) - exp(-ka*t))
    # Using normalized Vd=1 for relative intensity
    return (dose * ka / (ka - 0.001 if ka == ke else ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

# Helper to fix NumPy version issues
def calculate_area(y, x):
    try:
        return np.trapezoid(y, x) # NumPy 2.0+
    except AttributeError:
        return np.trapz(y, x) # NumPy < 2.0

# --- 2. DATASET: ABILIFY & COUNTER-MEDS ---
ABILIFY_DATA = {
    "ka": 1.0, 
    "ke": np.log(2) / 75, # 75h half-life
    "dose": 100,
    "side_effects": {
        "Akathisia (Restlessness)": {"lag": 2.5, "color": "#e74c3c", "type": "neurological"},
        "Insomnia": {"lag": 4.0, "color": "#f39c12", "type": "metabolic"},
        "Nausea": {"lag": 0.5, "color": "#9b59b6", "type": "direct"},
        "Dizziness": {"lag": 1.0, "color": "#d35400", "type": "direct"},
        "Fatigue/Somnolence": {"lag": 5.0, "color": "#34495e", "type": "metabolic"},
        "Blurred Vision": {"lag": 2.0, "color": "#7f8c8d", "type": "direct"},
        "Anxiety": {"lag": 1.5, "color": "#ff6b6b", "type": "neurological"},
        "Tremor": {"lag": 2.5, "color": "#c0392b", "type": "neurological"},
        "Dry Mouth": {"lag": 3.0, "color": "#1abc9c", "type": "direct"},
        "Headache": {"lag": 2.0, "color": "#2c3e50", "type": "direct"}
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
st.set_page_config(page_title="Med-Engineering AI", layout="wide")
st.title("💊 Clinical Decision Support System: Med-Engineering")
st.markdown("---")

col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Primary Med
    st.subheader("1. Primary Medication")
    med_select = st.selectbox("Medicine", ["Abilify (Aripiprazole)"])
    
    # 2. Timing
    st.subheader("2. Schedule")
    dose_date = st.date_input("Dose Date", datetime.now())
    dose_time = st.time_input("Dose Time", value=datetime.strptime("10:00 PM", "%I:%M %p").time())
    dt_dose = datetime.combine(dose_date, dose_time)
    
    # 3. Side Effect
    st.subheader("3. Logic Kernel")
    se_name = st.selectbox("Select Symptom", list(ABILIFY_DATA["side_effects"].keys()))
    
    # 4. Counter Med
    st.subheader("4. Mitigation Strategy")
    counter_select = st.selectbox("Counter Medication", list(COUNTER_MEDS.keys()))

with col_main:
    # --- 4. CALCULATIONS ---
    h_axis = np.linspace(0, 48, 1000) # 48 hour simulation
    
    # Abilify Curve
    abilify_curve = pk_model(h_axis, ABILIFY_DATA["dose"], ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])
    
    # Side Effect Curve (using the Lag Logic)
    se_data = ABILIFY_DATA["side_effects"][se_name]
    se_curve = pk_model(h_axis - se_data["lag"], ABILIFY_DATA["dose"] * 0.85, ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])
    
    # Recommender Logic
    peak_idx = np.argmax(se_curve)
    peak_h = h_axis[peak_idx]
    peak_time_dt = dt_dose + timedelta(hours=peak_h)
    
    # Counter Med Curve
    counter_curve = np.zeros_like(h_axis)
    rec_time_str = "N/A"
    coverage = 0
    
    if counter_select != "None":
        c_params = COUNTER_MEDS[counter_select]
        # Align peaks: Dose counter-med such that its peak hits at the side effect's peak
        opt_offset = peak_h - c_params["t_max"]
        counter_curve = pk_model(h_axis - opt_offset, 115, c_params["ka"], c_params["ke"])
        rec_dt = dt_dose + timedelta(hours=opt_offset)
        rec_time_str = rec_dt.strftime("%m/%d %I:%M %p")
        
        # Calculate Mitigation Coverage (Yellow Area)
        mitigated_area = calculate_area(np.minimum(se_curve, counter_curve), h_axis)
        total_se_area = calculate_area(se_curve, h_axis)
        coverage = (mitigated_area / total_se_area) * 100 if total_se_area > 0 else 0

    # Metrics Display
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Predicted Peak Intensity", peak_time_dt.strftime("%I:%M %p"))
    m_col2.metric("Recommended Counter-Dose", rec_time_str)
    m_col3.metric("Symptom Mitigation Coverage", f"{coverage:.1f}%")

    # --- 5. VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h_axis, abilify_curve, label="Abilify Plasma Conc.", color="#3498db", alpha=0.3, lw=1)
    ax.plot(h_axis, se_curve, label=f"Risk: {se_name}", color=se_data["color"], lw=3)
    
    if counter_select != "None":
        ax.plot(h_axis, counter_curve, label=f"Shield: {counter_select}", color="#27ae60", lw=2)
        mitigation = np.minimum(se_curve, counter_curve)
        ax.fill_between(h_axis, 0, mitigation, color="#f1c40f", alpha=0.4, label="Mitigation Zone")

    # Formatting
    ax.set_title(f"Dynamic Modeling: {se_name} Mitigation", fontsize=14)
    ax.set_xlabel("Hours Since First Dose")
    ax.set_ylabel("Clinical Intensity")
    ax.legend(frameon=True, loc='upper right')
    ax.grid(alpha=0.2, ls='--')
    
    st.pyplot(fig)
    
    st.info(f"**Insight:** To neutralize the peak of {se_name}, the system recommends administering {counter_select} at **{rec_time_str}** to align the metabolic peaks perfectly.")
