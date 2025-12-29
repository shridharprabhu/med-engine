import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from supabase import create_client, Client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

supabase = get_supabase()

def login_ui():
    st.title("Clinic Dashboard Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            auth_res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["sb_session"] = auth_res.session
            st.session_state["sb_user"] = auth_res.user
            st.success("Logged in!")
            st.rerun()
        except Exception as e:
            st.error("Login failed.")
            st.caption(str(e))

def logout_button():
    if st.sidebar.button("Log out"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

def fetch_profile(user_id: str, access_token: str):
    authed = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    authed.postgrest.auth(access_token)

    res = (
        authed.table("profiles")
        .select("user_id, clinic_id, role, full_name")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return res.data

def require_auth():
    sess = st.session_state.get("sb_session")
    user = st.session_state.get("sb_user")

    if not sess or not user:
        login_ui()
        st.stop()

    # Load profile (clinic_id + role)
    try:
        profile = fetch_profile(user.id, sess.access_token)
        st.session_state["profile"] = profile
    except Exception as e:
        st.error("Logged in, but profile not found / not accessible.")
        st.caption("Fix: Insert a row in public.profiles for this user in Supabase.")
        st.caption(str(e))
        st.stop()



require_auth()
logout_button()


st.sidebar.success(f"Logged in as: {st.session_state['sb_user'].email}")
st.sidebar.write(st.session_state["profile"])



# ----------------------------
# 0) CONFIG
# ----------------------------
st.set_page_config(page_title="Chronopharmacology Dashboard", layout="wide")

CSS = """
<style>
  :root { --card:#111827; --stroke:#243244; --accent:#60a5fa; --muted: rgba(255,255,255,0.72); }
  .stApp { background-color: #0e1117; color: white; }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #0e1117 60%); }
  .block-container { padding-top: 1.35rem; padding-bottom: 1.35rem; }

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

  label { color: rgba(255,255,255,0.82) !important; }

  /* Small badge */
  .pill {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid rgba(36,50,68,0.9);
    background: rgba(17,24,39,0.75);
    font-weight: 700;
    letter-spacing: 0.2px;
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("🛡️ Chronopharmacology Dashboard")
st.info(
    "Prototype: side-effect list is derived from FDA openFDA FAERS frequency counts; "
    "interaction rating is derived from FDA drug labeling text. Visualization-only; not medical advice."
)

# ----------------------------
# 1) PK MODEL
# ----------------------------
def pk_model(t_hours: np.ndarray, dose: float, ka: float, ke: float) -> np.ndarray:
    t = np.maximum(t_hours, 0.0)
    if abs(ka - ke) < 1e-9:
        ka = ka + 1e-6
    return (dose * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))

def normalize(x: np.ndarray) -> np.ndarray:
    mx = float(np.max(x)) if np.max(x) > 0 else 1.0
    return x / mx

# ----------------------------
# 2) MED LIST (20 COMMON PSYCH MEDS)
# ----------------------------
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

# Display -> openFDA generic search token (best-effort)
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

# Pragmatic PK shaping (replace later with your extracted values)
MED_HALF_LIFE_HOURS = {
    "aripiprazole": 75,
    "risperidone": 20,
    "olanzapine": 30,
    "quetiapine": 6,
    "ziprasidone": 7,
    "clozapine": 12,
    "haloperidol": 20,
    "fluoxetine": 96,
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
    "default": 1.2,
    "fluoxetine": 0.9,
    "aripiprazole": 1.0,
    "quetiapine": 2.0,
    "venlafaxine": 2.0,
}

# ----------------------------
# 3) COUNTER MEDS (SAME AS YOUR LIST)
# ----------------------------
COUNTER_MEDS = {
    "None": None,
    "Clonazepam": {"ka": 1.8, "ke": np.log(2) / 35, "t_max": 2.0, "generic": "clonazepam"},
    "Propranolol": {"ka": 2.5, "ke": np.log(2) / 4.0, "t_max": 1.5, "generic": "propranolol"},
    "Melatonin": {"ka": 3.0, "ke": np.log(2) / 1.0, "t_max": 0.5, "generic": "melatonin"},
    "Guanfacine": {"ka": 1.2, "ke": np.log(2) / 17, "t_max": 3.0, "generic": "guanfacine"},
    "Hydroxyzine": {"ka": 2.0, "ke": np.log(2) / 20, "t_max": 2.0, "generic": "hydroxyzine"},
    "Benadryl": {"ka": 2.2, "ke": np.log(2) / 8, "t_max": 2.0, "generic": "diphenhydramine"},
    "Zofran": {"ka": 2.8, "ke": np.log(2) / 3.5, "t_max": 1.5, "generic": "ondansetron"},
    "Trazodone": {"ka": 1.5, "ke": np.log(2) / 10, "t_max": 2.0, "generic": "trazodone"},
    "Magnesium": {"ka": 0.8, "ke": np.log(2) / 12, "t_max": 4.0, "generic": "magnesium"},
    "Xanax": {"ka": 3.0, "ke": np.log(2) / 11, "t_max": 1.0, "generic": "alprazolam"},
}

# ----------------------------
# 4) openFDA
# ----------------------------
OPENFDA_EVENT = "https://api.fda.gov/drug/event.json"
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"

def _api_key() -> str | None:
    return st.secrets.get("OPENFDA_API_KEY", None) if hasattr(st, "secrets") else None

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_top_reactions_for_med(generic_name: str, limit: int = 12) -> list[str]:
    """
    FAERS frequency (count) of reactions for a drug search.
    """
    query = f'patient.drug.openfda.generic_name:"{generic_name}"'
    params = {"search": query, "count": "patient.reaction.reactionmeddrapt.exact", "limit": str(limit)}
    key = _api_key()
    if key:
        params["api_key"] = key

    try:
        r = requests.get(OPENFDA_EVENT, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        terms = [row["term"].title().strip() for row in data.get("results", []) if row.get("term")]
        return terms[:limit] if terms else ["No data found"]
    except Exception:
        return ["No data found"]

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_label_text_bundle(generic_name: str) -> dict:
    """
    Fetch a small set of label sections that are relevant to interactions.
    openFDA label fields include drug_interactions, contraindications, warnings, boxed_warning, etc.
    We query by openfda.generic_name where possible.
    """
    query = f'openfda.generic_name:"{generic_name}"'
    params = {"search": query, "limit": "1"}
    key = _api_key()
    if key:
        params["api_key"] = key

    bundle = {
        "generic": generic_name,
        "drug_interactions": "",
        "contraindications": "",
        "warnings": "",
        "boxed_warning": "",
        "warnings_and_precautions": "",
        "precautions": "",
        "success": False,
    }

    try:
        r = requests.get(OPENFDA_LABEL, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        res = (data.get("results") or [])
        if not res:
            return bundle

        rec = res[0]

        def join_field(field: str) -> str:
            v = rec.get(field, "")
            if isinstance(v, list):
                return "\n".join([str(x) for x in v if x])
            if isinstance(v, str):
                return v
            return ""

        bundle["drug_interactions"] = join_field("drug_interactions")
        bundle["contraindications"] = join_field("contraindications")
        bundle["warnings"] = join_field("warnings")
        bundle["boxed_warning"] = join_field("boxed_warning")
        bundle["warnings_and_precautions"] = join_field("warnings_and_precautions")
        bundle["precautions"] = join_field("precautions")
        bundle["success"] = True
        return bundle
    except Exception:
        return bundle

def reaction_lag_hours(reaction: str, med_half_life: float) -> float:
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

    scale = 1.0 + 0.25 * np.log(max(med_half_life, 6) / 6.0)
    return float(np.clip(base * scale, 0.3, 10.0))

# ----------------------------
# 5) INTERACTION SEVERITY ENGINE (label-text based)
# ----------------------------
def interaction_severity(main_generic: str, counter_generic: str) -> tuple[str, str]:
    """
    Returns (severity, evidence_snippet).

    Heuristic:
    - If either label's contraindications mention the other drug -> SEVERE
    - If drug_interactions mention the other + "avoid", "do not", "contraindicated" -> HIGH/SEVERE
    - If interactions mention the other + "monitor", "dose adjust", "caution" -> MED
    - If no mention -> LOW (unknown/none detected in label text)
    """
    if not counter_generic or counter_generic.strip().lower() == "none":
        return ("low", "No counter medication selected.")

    main = fetch_label_text_bundle(main_generic)
    ctr = fetch_label_text_bundle(counter_generic)

    # Combine “high-signal” sections
    main_text = "\n\n".join([
        main.get("boxed_warning", ""),
        main.get("contraindications", ""),
        main.get("warnings", ""),
        main.get("warnings_and_precautions", ""),
        main.get("drug_interactions", ""),
        main.get("precautions", ""),
    ]).lower()

    ctr_text = "\n\n".join([
        ctr.get("boxed_warning", ""),
        ctr.get("contraindications", ""),
        ctr.get("warnings", ""),
        ctr.get("warnings_and_precautions", ""),
        ctr.get("drug_interactions", ""),
        ctr.get("precautions", ""),
    ]).lower()

    target = counter_generic.lower()
    target2 = main_generic.lower()

    def find_snippet(text: str, needle: str, window: int = 220) -> str:
        idx = text.find(needle)
        if idx == -1:
            return ""
        start = max(0, idx - window // 2)
        end = min(len(text), idx + window // 2)
        snip = text[start:end].strip()
        snip = " ".join(snip.split())
        return snip

    # Check if either label explicitly names the other drug
    main_mentions = (target in main_text)
    ctr_mentions = (target2 in ctr_text)

    # If no label text found (common for supplements), degrade gracefully
    if not main.get("success") and not ctr.get("success"):
        return ("low", "No labeling text retrieved for either drug via openFDA; interaction not detected from label data.")

    # Strong signals
    severe_words = ["contraindicated", "do not use", "do not coadminister", "do not administer", "avoid concomitant"]
    high_words = ["avoid", "do not", "not recommended", "serious", "life-threatening", "torsades", "arrhythmia", "qt prolongation"]
    med_words = ["monitor", "caution", "dose adjustment", "reduce dose", "increase dose", "may increase", "may decrease", "consider"]

    # Evaluate evidence in priority order: contraindications first
    evidence = ""
    severity = "low"

    # If main contraindications mention counter
    if main_mentions and target in (main.get("contraindications", "").lower()):
        evidence = find_snippet(main.get("contraindications", "").lower(), target) or "Mentioned in contraindications (main drug label)."
        return ("severe", evidence)

    # If counter contraindications mention main
    if ctr_mentions and target2 in (ctr.get("contraindications", "").lower()):
        evidence = find_snippet(ctr.get("contraindications", "").lower(), target2) or "Mentioned in contraindications (counter drug label)."
        return ("severe", evidence)

    # Otherwise scan around the mention in drug_interactions / warnings
    if main_mentions:
        # Prefer drug_interactions
        sect = (main.get("drug_interactions", "") or main_text).lower()
        snip = find_snippet(sect, target) or find_snippet(main_text, target)
        evidence = snip

        # Keyword severity
        if any(w in snip for w in severe_words):
            severity = "severe"
        elif any(w in snip for w in high_words):
            severity = "high"
        elif any(w in snip for w in med_words):
            severity = "med"
        else:
            severity = "med"  # mention exists but no clear language; treat as medium caution

    if ctr_mentions and severity == "low":
        sect = (ctr.get("drug_interactions", "") or ctr_text).lower()
        snip = find_snippet(sect, target2) or find_snippet(ctr_text, target2)
        evidence = snip

        if any(w in snip for w in severe_words):
            severity = "severe"
        elif any(w in snip for w in high_words):
            severity = "high"
        elif any(w in snip for w in med_words):
            severity = "med"
        else:
            severity = "med"

    # If neither label mentions the other, keep low
    if not main_mentions and not ctr_mentions:
        return ("low", "No explicit mention of co-administration found in openFDA label sections queried.")

    return (severity, evidence or "Interaction mention detected in label text; no clear severity language found.")

def severity_badge(sev: str) -> str:
    sev = (sev or "low").lower()
    if sev == "severe":
        return '<span class="pill" style="border-color: rgba(248,113,113,0.75); color:#fecaca;">severe</span>'
    if sev == "high":
        return '<span class="pill" style="border-color: rgba(251,146,60,0.75); color:#fed7aa;">high</span>'
    if sev == "med":
        return '<span class="pill" style="border-color: rgba(250,204,21,0.75); color:#fef08a;">med</span>'
    return '<span class="pill" style="border-color: rgba(34,197,94,0.75); color:#bbf7d0;">low</span>'

# ----------------------------
# 6) UI INPUTS
# ----------------------------
with st.sidebar:
    st.header("📋 Clinical Inputs")

    selected_med = st.selectbox("Selected medication", PSYCH_MEDS)
    med_search_name = MED_SEARCH_NAME[selected_med]

    med_dose_date = st.date_input("Medication dose date", date(2025, 12, 9), key="med_date")
    med_dose_time = st.time_input("Medication dose time", datetime.strptime("10:00 PM", "%I:%M %p").time(), key="med_time")

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

half_life = float(MED_HALF_LIFE_HOURS.get(med_search_name, 24))
ke_med = np.log(2) / half_life
ka_med = float(MED_KA.get(med_search_name, MED_KA["default"]))
dose_med = 100.0

med_curve = normalize(pk_model(t_plot, dose_med, ka_med, ke_med))

lag = reaction_lag_hours(selected_se, half_life)
se_curve = normalize(pk_model(t_plot - lag, 80.0, ka_med, ke_med))

# onset = 15% of peak
se_threshold = float(np.max(se_curve)) * 0.15
onset_idx = np.where(se_curve > se_threshold)[0]
se_onset_hour = float(t_plot[onset_idx[0]]) if len(onset_idx) else 0.0

se_peak_hour = float(t_plot[np.argmax(se_curve)])
se_peak_time = dt_med + timedelta(hours=se_peak_hour)
se_develop_time = dt_med + timedelta(hours=se_onset_hour)

counter_curve = np.zeros_like(t_plot)
optimal_counter_time = None
optimal_counter_offset = None

if counter_med != "None":
    c = COUNTER_MEDS[counter_med]
    optimal_counter_offset = max(se_onset_hour - float(c["t_max"]), 0.0)
    optimal_counter_time = dt_med + timedelta(hours=optimal_counter_offset)

    user_offset = (dt_counter_user - dt_med).total_seconds() / 3600.0
    counter_curve = normalize(pk_model(t_plot - user_offset, 110.0, float(c["ka"]), float(c["ke"])))

covered = np.minimum(se_curve, counter_curve) if counter_med != "None" else np.zeros_like(se_curve)

# ----------------------------
# 8) DRUG INTERACTION SECTION
# ----------------------------
counter_generic = None
if counter_med != "None":
    counter_generic = COUNTER_MEDS[counter_med]["generic"]
else:
    counter_generic = "none"

sev, evidence = interaction_severity(med_search_name, counter_generic)

# ----------------------------
# 9) METRICS
# ----------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Side Effect Peak Predicted", se_peak_time.strftime("%m/%d %I:%M %p"))
with m2:
    st.metric("Side Effect Develops ~", se_develop_time.strftime("%m/%d %I:%M %p"))
with m3:
    if counter_med == "None":
        st.metric("Optimal Counter-Dose Time", "None selected")
    else:
        st.metric("Optimal Counter-Dose Time", optimal_counter_time.strftime("%m/%d %I:%M %p"))
with m4:
    # Display severity as a compact pill
    st.markdown("**Drug Interaction Risk**", unsafe_allow_html=False)
    st.markdown(severity_badge(sev), unsafe_allow_html=True)

with st.expander("Drug interaction evidence (from FDA label sections queried)"):
    st.write(evidence if evidence else "No evidence snippet available.")

# ----------------------------
# 10) PLOT
# ----------------------------
fig, ax = plt.subplots(figsize=(12.8, 5.6), facecolor="#0e1117")
ax.set_facecolor("#0e1117")

ax.tick_params(colors="white", which="both", labelsize=10)
for spine in ax.spines.values():
    spine.set_color("#2b3443")

ax.plot(t_plot, med_curve, label="Medication curve", color="#60a5fa", lw=2.2, alpha=0.78)
ax.plot(t_plot, se_curve, label=f"Side effect risk: {selected_se}", color="#fb7185", lw=2.7, alpha=0.95)

if counter_med != "None":
    ax.plot(t_plot, counter_curve, label=f"Counter-med: {counter_med}", color="#22c55e", lw=2.6, alpha=0.95)

    # Green = covered by counter, light red = uncovered
    ax.fill_between(t_plot, 0, covered, color="#22c55e", alpha=0.18, label="Covered by counter-med")
    ax.fill_between(t_plot, covered, se_curve, color="#fb7185", alpha=0.14, label="Uncovered risk")

    if optimal_counter_offset is not None:
        ax.axvline(optimal_counter_offset, color="#22c55e", alpha=0.22, lw=1.3, linestyle="--")

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
# 11) FOOTNOTE / DISCLOSURE
# ----------------------------
st.caption(
    "FAERS limitation: reports do not establish causality; and when multiple drugs/reactions are present, "
    "no individual drug is connected to an individual reaction. "
    "Drug interaction risk here is a heuristic scan of openFDA label sections (Drug Interactions / Contraindications / Warnings)."
)
