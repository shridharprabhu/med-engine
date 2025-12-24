import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Med-Engineer SaaS", layout="wide")
st.title("🧠 Clinical Decision Support: Chrono-Optimization")

# --- DATA DICTIONARY (The Intelligence Layer) ---
# In a full scale app, this would be a JSON or Database
SIDE_EFFECT_DB = {
    "Crash (Withdrawal/Rebound)": {"category": 2, "description": "Symptoms peak during rapid drug decline."},
    "Akathisia (Restlessness)": {"category": 1, "lag": 1.5, "description": "Follows blood concentration with a neural lag."},
    "Nausea (Peak Toxicity)": {"category": 1, "lag": 0.5, "description": "Linked to rapid rise in plasma levels."},
    "Brain Fog (Cumulative)": {"category": 3, "description": "Driven by total daily exposure (AUC)."}
}

# --- CORE MATH ENGINE ---
def pk_model(t_hours, ka, ke):
    t = np.maximum(t_hours, 0)
    return (1.0 * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(curve, target_max=100):
    if np.max(curve) == 0: return curve
    return (curve / np.max(curve)) * target_max

def parse_med_time(dt_str):
    dt_str = dt_str.lower().strip()
    full_str = f"{dt_str} 2025" 
    for fmt in ["%m/%d %I%p %Y", "%m/%d %I:%M%p %Y"]:
        try: return datetime.strptime(full_str, fmt)
        except ValueError: continue
    return None

# --- SIDEBAR & DROPDOWN ---
st.sidebar.header("Patient Regimen")
main_med_time = st.sidebar.text_input("Main Med (e.g. Adderall) MM/DD Time:", "12/24 8am")
counter_med_time = st.sidebar.text_input("Counter Med (e.g. Clonazepam) MM/DD Time:", "12/24 2pm")

selected_se = st.sidebar.selectbox("Target Side Effect to Mitigate:", list(SIDE_EFFECT_DB.keys()))

# --- EXECUTION ---
dt1 = parse_med_time(main_med_time)
dt2 = parse_med_time(counter_med_time)

if dt1 and dt2:
    start_plot = dt1 - timedelta(hours=2)
    h_axis = np.linspace(0, 36, 1000)
    t_since_1 = np.array([((start_plot + timedelta(hours=h)) - dt1).total_seconds()/3600 for h in h_axis])
    t_since_2 = np.array([((start_plot + timedelta(hours=h)) - dt2).total_seconds()/3600 for h in h_axis])

    # Main Med (Generic Stimulant/Agonist Parameters)
    conc_1 = pk_model(t_since_1, 1.1, np.log(2)/10)
    conc_1_norm = normalize(conc_1, 100)

    # DYNAMIC SIDE EFFECT LOGIC
    se_logic = SIDE_EFFECT_DB[selected_se]
    if se_logic["category"] == 1:
        # Lagged Peak
        se_curve = pk_model(t_since_1 - se_logic["lag"], 1.1, np.log(2)/10)
        se_norm = normalize(se_curve, 90)
    elif se_logic["category"] == 2:
        # Rebound (Crash) - Triggered at <50% of peak
        dC_dt = np.gradient(conc_1_norm, h_axis)
        se_trigger = np.where((dC_dt < 0) & (conc_1_norm < 50), np.abs(dC_dt), 0)
        se_norm = normalize(se_trigger, 90)
    else: # Category 3 - Cumulative
        se_norm = normalize(np.cumsum(conc_1_norm), 90)

    # Counter Med
    conc_2 = pk_model(t_since_2, 1.8, np.log(2)/35)
    conc_2_norm = normalize(conc_2, 100)

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(h_axis, conc_1_norm, color='gray', alpha=0.4, label='Main Med Concentration')
    ax.plot(h_axis, se_norm, color='red', ls='--', lw=2, label=f'Predicted {selected_se}')
    ax.plot(h_axis, conc_2_norm, color='green', lw=2, label='Counter Med (Neutralizer)')

    mitigation = np.minimum(se_norm, conc_2_norm)
    ax.fill_between(h_axis, 0, mitigation, color='gold', alpha=0.4, label='Neutralization Window')

    # Formatting
    tick_indices = np.arange(0, 37, 4)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([(start_plot + timedelta(hours=int(h))).strftime("%m/%d\n%I%p") for h in tick_indices])
    ax.legend()
    st.pyplot(fig)

    # --- 2. THE RECOMMENDER SYSTEM (The Sauce) ---
    st.divider()
    st.subheader("💡 Chrono-Optimizer Recommendation")
    
    # Logic: Find the time index where the Side Effect peaks
    se_peak_idx = np.argmax(se_norm)
    se_peak_time = start_plot + timedelta(hours=h_axis[se_peak_idx])
    
    # Logic: Counter Med T-max (peak) is approx 2.5 hours
    # To neutralize, the counter-med peak should hit at the same time as the SE peak
    optimal_dose_time = se_peak_time - timedelta(hours=2.5)
    
    # Calculate current effectiveness
    current_effectiveness = np.max(mitigation)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Side Effect Peak Predicted At:", se_peak_time.strftime("%I:%M %p"))
        st.write(f"**Current Strategy Effectiveness:** {int(current_effectiveness)}%")

    with col2:
        st.write("### Recommended Adjustment:")
        st.info(f"To maximize relief, the counter-medication should be administered at **{optimal_dose_time.strftime('%I:%M %p')}**.")
        
    st.caption("Calculated based on T-max peak alignment and metabolic rate of decline.")

else:
    st.info("Awaiting valid date/time inputs to generate clinical map.")
