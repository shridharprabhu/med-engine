import streamlit as st
import numpy as np
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

# IMPORTANT:
# - st.markdown defaults to rendering Markdown safely.
# - To apply CSS, we must set unsafe_allow_html=True.
# If your Streamlit runtime is somehow injecting unsafe_allow_stdio, we avoid *all* kwargs by
# using st.html where available, otherwise fall back to minimal UI without CSS.
#
# Streamlit versions differ; st.html may not exist in older versions. We handle both.

CSS = """
<style>
  :root { --card:#111827; --stroke:#243244; --accent:#60a5fa; }
  .stApp { background-color: #0e1117; color: white; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #0e1117 60%); }
  .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }

  div[data-testid="stMetric"] {
    background: rgba(17,24,39,0.85);
    border: 1px solid rgba(36,50,68,0.9);
    padding: 14px 16px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  div[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.72); }
  div[data-testid="stMetricValue"] { font-size: 1.85rem; color: var(--accent); }

  div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(36,50,68,0.9);
    background: rgba(17,24,39,0.55);
  }
</style>
"""

# Try to inject CSS without using any suspicious kwargs.
# 1) Prefer st.html if available
# 2) Else use st.markdown with explicit unsafe_allow_html=True (correct Streamlit param)
# If your deployment is somehow rewriting kwargs, st.html path avoids that.
if hasattr(st, "html"):
    st.html(CSS)
else:
    st.markdown(CSS, unsafe_allow_html=True)

st.title("🛡️ Chronopharmacology Dashboard")
st.info("Precision Side-Effect Engineering for Aripiprazole (Abilify)")

with st.sidebar:
    st.header("📋 Clinical Inputs")
    dose_date = st.date_input("Dose Date", datetime(2025, 12, 9))
    dose_time = st.time_input(
        "Dose Time",
        value=datetime.strptime("10:00 PM", "%I:%M %p").time()
    )
    selected_se = st.selectbox("Observed Side Effect", list(ABILIFY_DATA["side_effects"].keys()))
    counter_med = st.selectbox("Select Counter Medication", list(COUNTER_MEDS.keys()))

# --- 4. DATA PROCESSING ---
dt_dose = datetime.combine(dose_date, dose_time)
t_plot = np.linspace(0, 48, 1000)

abilify_conc = pk_model(t_plot, ABILIFY_DATA["dose"], ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

se_info = ABILIFY_DATA["side_effects"][selected_se]
se_curve = pk_model(t_plot - se_info["lag"], 80, ABILIFY_DATA["ka"], ABILIFY_DATA["ke"])

# Side effect "develops" when it crosses 15% of its peak
se_threshold = np.max(se_curve) * 0.15
onset_indices = np.where(se_curve > se_threshold)[0]
se_onset_hour = float(t_plot[onset_indices[0]]) if len(onset_indices) > 0 else 0.0

predicted_peak_time = dt_dose + timedelta(hours=float(t_plot[np.argmax(se_curve)]))

counter_conc = np.zeros_like(t_plot)
rec_time_str = "None Selected"
optimal_offset = None

if counter_med != "None":
    c_data = COUNTER_MEDS[counter_med]

    # ✅ Fine-tuned recommendation:
    # Take counter-med so its peak lands right before side-effect development.
    optimal_offset = max(se_onset_hour - float(c_data["t_max"]), 0.0)

    counter_conc = pk_model(t_plot - optimal_offset, 110, c_data["ka"], c_data["ke"])
    rec_time = dt_dose + timedelta(hours=optimal_offset)
    rec_time_str = rec_time.strftime("%m/%d %I:%M %p")

# --- 5. DASHBOARD METRICS ---
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Side Effect Peak", predicted_peak_time.strftime("%I:%M %p"))
with m2:
    st.metric("Side Effect Develops ~", (dt_dose + timedelta(hours=se_onset_hour)).strftime("%I:%M %p"))
with m3:
    st.metric("Optimal Counter-Dose Time", rec_time_str)

# --- 6. MODERN VISUALIZATION ---
fig, ax = plt.subplots(figsize=(12.5, 5.4), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

ax.tick_params(colors="white", which="both", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#2b3443")

ax.plot(t_plot, abilify_conc, label="Abilify Conc.", color="#60a5fa", alpha=0.30, lw=1.25)
ax.plot(t_plot, se_curve, label=f"{selected_se} (Side Effect Risk)", color=se_info["color"], lw=2.7, alpha=0.95)

if counter_med != "None":
    ax.plot(t_plot, counter_conc, label=f"Counter: {counter_med}", color="#22c55e", lw=2.5, alpha=0.95)

    # ✅ Green covered area vs light red uncovered area
    covered = np.minimum(se_curve, counter_conc)
    ax.fill_between(t_plot, 0, covered, color="#22c55e", alpha=0.18, label="Covered by Counter-med")
    ax.fill_between(t_plot, covered, se_curve, color="#fb7185", alpha=0.18, label="Uncovered Risk")

    # Subtle marker at recommended time
    ax.axvline(optimal_offset, color="#22c55e", alpha=0.25, lw=1.2, linestyle="--")

# ✅ Clean x-axis labels (two-line, horizontal)
tick_hours = np.arange(0, 49, 6)
time_labels = [
    (dt_dose + timedelta(hours=int(h))).strftime("%m/%d") + "\n" +
    (dt_dose + timedelta(hours=int(h))).strftime("%I:%M %p")
    for h in tick_hours
]
ax.set_xticks(tick_hours)
ax.set_xticklabels(time_labels, color="white", rotation=0, ha="center")

# ✅ Y-axis labels horizontal
ax.tick_params(axis="y", labelrotation=0)
ax.set_ylabel("Clinical Intensity", color="white", fontsize=12)

ax.grid(color="#2b3443", linestyle="--", alpha=0.35)
ax.legend(facecolor="#111827", edgecolor="#2b3443", labelcolor="white", loc="upper right", framealpha=0.9)

fig.subplots_adjust(bottom=0.22, left=0.07, right=0.98, top=0.95)
st.pyplot(fig)
