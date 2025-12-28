import os
import re
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date, time

# ----------------------------
# 0) CONFIG
# ----------------------------
st.set_page_config(page_title="Chronopharmacology Dashboard", layout="wide")

CSS = """
<style>
  :root { --card:#111827; --stroke:#243244; --accent:#60a5fa; --muted: rgba(255,255,255,0.72); }
  .stApp { background-color: #0e1117; color: white; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #0e1117 60%); }
  .block-container { padding-top: 1.4rem; padding-bottom: 1.4rem; }

  /* Metric cards */
  div[data-testid="stMetric"] {
    background: rgba(17,24,39,0.85);
    border: 1px solid rgba(36,50,68,0.9);
    padding: 14px 16px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  div[data-testid="stMetricLabel"] { color: var(--muted); }
  div[data-testid="stMetricValue"] { font-size: 1.85rem; color: var(--accent); }

  /* Alerts */
  div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(36,50,68,0.9);
    background: rgba(17,24,39,0.55);
  }

  /* Inputs */
  label { color: rgba(255,255,255,0.82) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("🛡️ Chronopharmacology Dashboard")
st.info("Prototype: data-driven side-effect list via FDA openFDA (FAERS). Visualization-only; not medical advice.")

# ----------------------------
# 1) PK MODEL
# ----------------------------
def pk_model(t_hours: np.ndarray, dose: float, ka: float, ke: float) -> np.ndarray:
    """
    1-compartment, first-order absorption and elimination.
    Returns a relative concentration curve (unitless).
    """
    t = np.maximum(t_hours, 0.0)
    if abs(ka - ke) < 1e-9:
        # Avoid division by zero (rare). Fallback to tiny perturbation.
        ka = ka + 1e-6
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x)) if np.max(x) > 0 else 1.0
    return x / m

# ----------------------------
# 2) MED LIST (20 COMMON PSYCH MEDS)
#    + a pragmatic PK parameter map
# ----------------------------
# NOTE: openFDA label data is largely free-text; half-life isn't reliably structured.
# We provide a reasonable half-life map for PK shape; you can later replace with your FDA-extracted values.
# This keeps your "custom curve per med" behavior consistent and deterministic.

PSYCH_MEDS = [
    "Aripiprazole (Abilify)",
    "Risperidone (Risperdal)",
    "Olanzapine (Zyprexa)",
    "Quetiapine (Seroquel)",
    "Ziprasidone (Geodon)",
    "Clozapine (Clozaril)",
    "Haloperidol (Haldol)",
    "Fluoxetine (Prozac)",
    "Sertraline (Zoloft)",
    "Escitalopram (Lexapro)",
    "Citalopram (Celexa)",
    "Paroxetine (Paxil)",
    "Venlafaxine (Effexor)",
    "Duloxetine (Cymbalta)",
    "Bupropion (Wellbutrin)",
    "Mirtazapine (Remeron)",
    "Lamotrigine (Lamictal)",
    "Lithium (Lithobid)",
    "Valproate (Depakote)",
    "Carbamazepine (Tegretol)",
]

# Map display name -> openFDA search name (generic preferred)
MED_SEARCH_NAME = {
    "Aripiprazole (Abilify)": "aripiprazole",
    "Risperidone (Risperdal)": "risperidone",
    "Olanzapine (Zyprexa)": "olanzapine",
    "Quetiapine (Seroquel)": "quetiapine",
    "Ziprasidone (Geodon)": "ziprasidone",
    "Clozapine (Clozaril)": "clozapine",
    "Haloperidol (Haldol)": "haloperidol",
    "Fluoxetine (Prozac)": "fluoxetine",
    "Sertraline (Zoloft)": "sertraline",
    "Escitalopram (Lexapro)": "escitalopram",
    "Citalopram (Celexa)": "citalopram",
    "Paroxetine (Paxil)": "paroxetine",
    "Venlafaxine (Effexor)": "venlafaxine",
    "Duloxetine (Cymbalta)": "duloxetine",
    "Bupropion (Wellbutrin)": "bupropion",
    "Mirtazapine (Remeron)": "mirtazapine",
    "Lamotrigine (Lamictal)": "lamotrigine",
    "Lithium (Lithobid)": "lithium",
    "Valproate (Depakote)": "valproic acid",
    "Carbamazepine (Tegretol)": "carbamazepine",
}

# Pragmatic PK shaping params (ka fixed-ish, ke derived from half-life)
# Replace these later with values you compute/extract per med.
MED_HALF_LIFE_HOURS = {
    "aripiprazole": 75,
    "risperidone": 20,
    "olanzapine": 30,
    "quetiapine": 6,
    "ziprasidone": 7,
    "clozapine": 12,
    "haloperidol": 20,
    "fluoxetine": 96,        # active metabolite longer; we keep shape long
    "sertraline": 26,
    "escitalopram": 27,
    "citalopram": 35,
    "paroxetine": 21,
    "venlafaxine": 5,
    "duloxetine": 12,
    "bupropion": 21,
    "mirtazapine": 30,
    "lamotrigine": 25,
    "lithium": 24,
    "valproic acid": 12,
    "carbamazepine": 15,
}

MED_KA = {
    # absorption rates (rough shape control)
    # higher ka = faster rise
    "default": 1.2,
    "fluoxetine": 0.9,
    "aripiprazole": 1.0,
    "quetiapine": 2.0,
    "venlafaxine": 2.0,
}

# ----------------------------
# 3) COUNTER MEDS (KEEP SAME SET AS YOU HAD)
# ----------------------------
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
    "Xanax": {"ka": 3.0, "ke": np.log(2) / 11, "t_max": 1.0},
}

# ----------------------------
# 4) openFDA FAERS: TOP SIDE EFFECTS PER MED
# ----------------------------
OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

def _api_key() -> str | None:
    # Streamlit Cloud: set in secrets as OPENFDA_API_KEY, or env var OPENFDA_API_KEY
    return st.secrets.get("OPENFDA_API_KEY", None) if hasattr(st, "secrets") else None

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_top_reactions_for_med(generic_name: str, limit: int = 12) -> list[str]:
    """
    Uses FAERS count endpoint to fetch top reactions for a drug by generic name.
    openFDA supports count queries on patient.reaction.reactionmeddrapt.exact. :contentReference[oaicite:1]{index=1}
    """
    # Prefer searching the harmonized openfda.generic_name; in practice you may need OR with brand name.
    # Keep query simple + robust.
    query = f'patient.drug.openfda.generic_name:"{generic_name}"'
    params = {
        "search": query,
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": str(limit),
    }
    key = _api_key()
    if key:
        params["api_key"] = key

    try:
        r = requests.get(OPENFDA_BASE, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        terms = [row["term"].title() for row in data.get("results", [])]
        # Make nicer, remove extremely generic noise if present
        cleaned = []
        for t in terms:
            t = t.strip()
            if not t:
                continue
            cleaned.append(t)
        return cleaned[:limit] if cleaned else ["No data found"]
    except Exception:
        return ["No data found"]

# ----------------------------
# 5) SIDE EFFECT CURVE SHAPING (custom per med + per reaction)
# ----------------------------
def reaction_lag_hours(reaction: str, med_half_life: float) -> float:
    """
    Heuristic lag:
    - Some reactions tend to be early (nausea, dizziness)
    - Others later (akathisia, insomnia, somnolence)
    Scaled mildly by half-life so long-half-life meds aren't unrealistically immediate.
    """
    r = reaction.lower()

    early = ["nausea", "vomiting", "dizziness", "headache", "dry mouth"]
    mid = ["anxiety", "agitation", "tremor", "restlessness", "akathisia"]
    late = ["insomnia", "somnolence", "fatigue", "sedation", "sleep"]

    base = 2.0
    if any(k in r for k in early):
        base = 0.8
    elif any(k in r for k in mid):
        base = 2.0
    elif any(k in r for k in late):
        base = 4.0

    # scale: half-life 6h -> ~1.0x, 75h -> ~1.6x
    scale = 1.0 + 0.25 * np.log(max(med_half_life, 6) / 6.0)
    return float(np.clip(base * scale, 0.3, 10.0))

# ----------------------------
# 6) UI INPUTS
# ----------------------------
with st.sidebar:
    st.header("📋 Clinical Inputs")

    selected_med = st.selectbox("Selected medication", PSYCH_MEDS)

    med_dose_date = st.date_input("Medication dose date", date(2025, 12, 9), key="med_date")
    med_dose_time = st.time_input("Medication dose time", datetime.strptime("10:00 PM", "%I:%M %p").time(), key="med_time")

    med_search_name = MED_SEARCH_NAME[selected_med]
    se_options = fetch_top_reactions_for_med(med_search_name, limit=12)
    selected_se = st.selectbox("Observed / target side effect", se_options)

    counter_med = st.selectbox("Counter medication", list(COUNTER_MEDS.keys()))

    counter_date = st.date_input("Counter-med dose date", med_dose_date, key="c_date")
    counter_time = st.time_input("Counter-med dose time", med_dose_time, key="c_time")

# ----------------------------
# 7) COMPUTE CURVES
# ----------------------------
dt_med = datetime.combine(med_dose_date, med_dose_time)
dt_counter_user = datetime.combine(counter_date, counter_time)

t_plot = np.linspace(0, 48, 1200)

# Med PK params
half_life = float(MED_HALF_LIFE_HOURS.get(med_search_name, 24))
ke_med = np.log(2) / half_life
ka_med = float(MED_KA.get(med_search_name, MED_KA["default"]))
dose_med = 100.0

med_curve = normalize(pk_model(t_plot, dose_med, ka_med, ke_med))

# Side effect curve = shifted med curve by lag (custom per reaction + med)
lag = reaction_lag_hours(selected_se, half_life)
se_curve = normalize(pk_model(t_plot - lag, 80.0, ka_med, ke_med))

# Onset definition: when SE crosses 15% of its peak
se_threshold = float(np.max(se_curve)) * 0.15
onset_idx = np.where(se_curve > se_threshold)[0]
se_onset_hour = float(t_plot[onset_idx[0]]) if len(onset_idx) else 0.0

se_peak_hour = float(t_plot[np.argmax(se_curve)])
se_peak_time = dt_med + timedelta(hours=se_peak_hour)
se_develop_time = dt_med + timedelta(hours=se_onset_hour)

# Counter curve: either user-chosen timing or "optimal recommender"
counter_curve = np.zeros_like(t_plot)
optimal_counter_time = None
optimal_counter_offset = None

if counter_med != "None":
    c = COUNTER_MEDS[counter_med]

    # Recommender: dose counter so it peaks right before side-effect development.
    # optimal_offset is relative to med dose time (dt_med).
    optimal_counter_offset = max(se_onset_hour - float(c["t_max"]), 0.0)
    optimal_counter_time = dt_med + timedelta(hours=optimal_counter_offset)

    # User-selected counter dose offset (relative to dt_med)
    user_offset = (dt_counter_user - dt_med).total_seconds() / 3600.0

    # Use user's selected counter timing to plot (what clinician chose),
    # while metrics show the recommended timing.
    counter_curve = normalize(pk_model(t_plot - user_offset, 110.0, float(c["ka"]), float(c["ke"])))

# Coverage shading
covered = np.minimum(se_curve, counter_curve) if counter_med != "None" else np.zeros_like(se_curve)
uncovered_top = se_curve

# ----------------------------
# 8) METRICS
# ----------------------------
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Side Effect Peak Predicted", se_peak_time.strftime("%m/%d %I:%M %p"))
with m2:
    st.metric("Side Effect Develops ~", se_develop_time.strftime("%m/%d %I:%M %p"))
with m3:
    if counter_med == "None":
        st.metric("Optimal Counter-Dose Time", "None selected")
    else:
        st.metric("Optimal Counter-Dose Time", optimal_counter_time.strftime("%m/%d %I:%M %p"))

# ----------------------------
# 9) PLOT
# ----------------------------
fig, ax = plt.subplots(figsize=(12.8, 5.6), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

ax.tick_params(colors="white", which="both", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#2b3443")

# Lines
ax.plot(t_plot, med_curve, label="Medication curve", color="#60a5fa", lw=2.2, alpha=0.75)
ax.plot(t_plot, se_curve, label=f"Side effect risk: {selected_se}", color="#fb7185", lw=2.7, alpha=0.95)

if counter_med != "None":
    ax.plot(t_plot, counter_curve, label=f"Counter-med: {counter_med}", color="#22c55e", lw=2.6, alpha=0.95)

    # Green where counter precedes/covers the SE curve
    ax.fill_between(t_plot, 0, covered, color="#22c55e", alpha=0.18, label="Covered by counter-med")

    # Light red where it doesn't
    ax.fill_between(t_plot, covered, uncovered_top, color="#fb7185", alpha=0.14, label="Uncovered risk")

    # Recommended time marker (subtle dashed)
    if optimal_counter_offset is not None:
        ax.axvline(optimal_counter_offset, color="#22c55e", alpha=0.22, lw=1.3, linestyle="--")

# Time axis labels (two-line, horizontal, readable)
tick_hours = np.arange(0, 49, 6)
xlabels = [
    (dt_med + timedelta(hours=int(h))).strftime("%m/%d") + "\n" +
    (dt_med + timedelta(hours=int(h))).strftime("%I:%M %p")
    for h in tick_hours
]
ax.set_xticks(tick_hours)
ax.set_xticklabels(xlabels, color="white", rotation=0, ha="center")

ax.tick_params(axis="y", labelrotation=0)
ax.set_ylabel("Relative clinical intensity (normalized)", color="white", fontsize=12)

ax.grid(color="#2b3443", linestyle="--", alpha=0.35)
ax.legend(facecolor="#111827", edgecolor="#2b3443", labelcolor="white", loc="upper right", framealpha=0.9)

fig.subplots_adjust(bottom=0.22, left=0.07, right=0.98, top=0.95)
st.pyplot(fig)

# ----------------------------
# 10) FOOTNOTE / DISCLOSURE
# ----------------------------
st.caption(
    "Data source: FDA openFDA FAERS (drug/event). FAERS is a spontaneous reporting system and does not establish causality. "
    "Reports can include multiple drugs and multiple reactions; the dataset does not connect a specific drug to a specific reaction. "
    "API limits apply; use an API key for higher daily limits. :contentReference[oaicite:2]{index=2}"
)
