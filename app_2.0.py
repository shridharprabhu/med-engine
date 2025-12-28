import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Safety + positioning
# -----------------------------
APP_TITLE = "Dashboard 2 — Medication Range Assistant (Label Reference + Risk Triage)"
DISCLAIMER = """
### Important safety note
This dashboard is **NOT** a dosing recommender and does **NOT** provide patient-specific dosage.
It provides:
- **FDA/label reference dosing text** (general reference)
- A **risk/caution classification** (low / med / high / severe) based on common contraindications/precautions
- A list of **triggered flags** to support clinician review

Always confirm using official labeling, local guidelines, and clinical judgment.
"""

# -----------------------------
# Label reference (non-personalized)
# Sources (for your documentation):
# - Clonazepam panic disorder dosing: DailyMed Klonopin (clonazepam) Dosage & Administration
# - Adderall IR dosing: FDA label (dextroamphetamine/amphetamine salts tablets)
# - Adderall XR contraindications + dosing: FDA label
# - Adderall XR renal note: not recommended ESRD in some labeling editions
# -----------------------------
LABEL_REFERENCE: Dict[str, Dict] = {
    "Clonazepam (Klonopin)": {
        "indications": {
            "Panic Disorder (Adults)": {
                "reference_dosing": [
                    "Initial: 0.25 mg twice daily",
                    "May increase to target 1 mg/day after 3 days",
                    "In fixed-dose study: 1 mg/day showed optimal effect; higher (2–4 mg/day) had more adverse effects",
                    "Some patients may benefit up to max 4 mg/day; increase by 0.125–0.25 mg twice daily every 3 days",
                    "Taper: decrease 0.125 mg twice daily every 3 days (example guidance)"
                ],
                "absolute_contra": [
                    "Significant liver disease (clinical or biochemical evidence)",
                    "History of sensitivity to benzodiazepines",
                    "Acute narrow-angle glaucoma"
                ]
            },
            "Seizure Disorders (Adults)": {
                "reference_dosing": [
                    "Initial: should not exceed 1.5 mg/day in 3 divided doses",
                    "Increase: 0.5–1 mg every 3 days until controlled or adverse effects limit",
                    "Max recommended daily dose: 20 mg/day (label maximum; individualized maintenance)"
                ],
                "absolute_contra": [
                    "Significant liver disease (clinical or biochemical evidence)",
                    "History of sensitivity to benzodiazepines",
                    "Acute narrow-angle glaucoma"
                ]
            }
        }
    },
    "Adderall IR (mixed amphetamine salts)": {
        "indications": {
            "ADHD (General label guidance)": {
                "reference_dosing": [
                    "Use the lowest effective dosage; adjust individually",
                    "Children 3–5: start 2.5 mg daily; increase by 2.5 mg weekly",
                    "Age ≥6: start 5 mg once or twice daily; increase by 5 mg weekly",
                    "Rarely necessary to exceed total 40 mg/day",
                    "Avoid late evening doses (insomnia)"
                ],
                "absolute_contra": [
                    "Moderate to severe hypertension",
                    "Symptomatic cardiovascular disease / serious heart disease",
                    "Advanced arteriosclerosis",
                    "Hyperthyroidism",
                    "Glaucoma",
                    "Agitated states",
                    "History of drug abuse",
                    "Use of MAOI within 14 days"
                ]
            }
        }
    },
    "Adderall XR (extended-release)": {
        "indications": {
            "ADHD (Label dosing)": {
                "reference_dosing": [
                    "Adults: 20 mg once daily in the morning (label statement)",
                    "Pediatric dosing varies by age; check label for full detail",
                    "Pretreatment screening: assess for cardiac disease (history/family history/exam)"
                ],
                "absolute_contra": [
                    "Moderate to severe hypertension",
                    "Symptomatic cardiovascular disease / serious heart disease",
                    "Advanced arteriosclerosis",
                    "Hyperthyroidism",
                    "Glaucoma",
                    "Known hypersensitivity to sympathomimetic amines",
                ],
                "renal_note": [
                    "Some labeling states: not recommended in ESRD (very low GFR)."
                ]
            }
        }
    }
}

# -----------------------------
# Risk model (triage, not dosing)
# low / med / high / severe
# -----------------------------
RISK_LEVELS = ["low", "med", "high", "severe"]

def escalate(level: str, new_level: str) -> str:
    return RISK_LEVELS[max(RISK_LEVELS.index(level), RISK_LEVELS.index(new_level))]

@dataclass
class PatientInputs:
    age: Optional[int]
    weight_kg: Optional[float]
    pregnancy: str
    sbp: Optional[int]
    dbp: Optional[int]
    heart_rate: Optional[int]
    cardiac_disease: str
    glaucoma: str
    hyperthyroidism: str
    liver_disease_known: str
    alt: Optional[float]
    ast: Optional[float]
    bilirubin: Optional[float]
    egfr: Optional[float]
    maoi_last_14d: str
    substance_use_disorder: str
    severe_anxiety_agitation: str
    sleep_severe_insomnia: str

def triage_clonazepam(p: PatientInputs) -> Tuple[str, List[str]]:
    level = "low"
    flags = []

    # Absolute contraindications / strong cautions from label
    if p.liver_disease_known == "Yes":
        level = escalate(level, "severe")
        flags.append("Known significant liver disease → clonazepam is contraindicated in significant liver disease (label).")

    if p.glaucoma == "Acute narrow-angle":
        level = escalate(level, "severe")
        flags.append("Acute narrow-angle glaucoma → clonazepam contraindicated (label).")

    # Lab-driven heuristic (triage-only)
    if p.alt is not None and p.alt >= 3 * 40:
        level = escalate(level, "high")
        flags.append("ALT very elevated (≥~3× ULN heuristic) → consider hepatic impairment risk; confirm with full workup.")
    if p.ast is not None and p.ast >= 3 * 40:
        level = escalate(level, "high")
        flags.append("AST very elevated (≥~3× ULN heuristic) → consider hepatic impairment risk; confirm with full workup.")
    if p.bilirubin is not None and p.bilirubin >= 2.0:
        level = escalate(level, "high")
        flags.append("Bilirubin elevated (heuristic) → consider hepatic dysfunction risk; confirm clinically.")

    # Age
    if p.age is not None and p.age >= 65:
        level = escalate(level, "med")
        flags.append("Older adult (≥65) → start low / monitor closely (general geriatric caution in label context).")

    # Pregnancy (triage only—no directive)
    if p.pregnancy != "No/Not applicable":
        level = escalate(level, "med")
        flags.append("Pregnancy/breastfeeding selected → requires specialist risk–benefit review.")

    return level, flags

def triage_adderall(p: PatientInputs) -> Tuple[str, List[str]]:
    level = "low"
    flags = []

    # Absolute contraindications from labeling summary
    if p.maoi_last_14d == "Yes":
        level = escalate(level, "severe")
        flags.append("MAOI in last 14 days → contraindication with amphetamines (label).")

    if p.glaucoma in ["Open-angle", "Acute narrow-angle"]:
        level = escalate(level, "severe")
        flags.append("Glaucoma selected → contraindication for amphetamines in labeling context.")

    if p.hyperthyroidism == "Yes":
        level = escalate(level, "severe")
        flags.append("Hyperthyroidism → contraindication (label).")

    if p.cardiac_disease == "Yes":
        level = escalate(level, "severe")
        flags.append("Known symptomatic cardiovascular disease/serious heart issues → contraindication / major warning area (label).")

    if p.substance_use_disorder == "Yes":
        level = escalate(level, "high")
        flags.append("History of substance use disorder → major misuse/addiction risk; requires tight controls (label warning context).")

    if p.severe_anxiety_agitation == "Yes":
        level = escalate(level, "high")
        flags.append("Severe anxiety/agitation selected → stimulants may worsen; labeling lists agitated states as a 'should not take' group.")

    # BP / HR heuristics for triage (not dosing)
    if p.sbp is not None and p.dbp is not None:
        if p.sbp >= 160 or p.dbp >= 100:
            level = escalate(level, "severe")
            flags.append("BP in moderate–severe range (heuristic) → labeling contraindicates moderate–severe hypertension.")
        elif p.sbp >= 140 or p.dbp >= 90:
            level = escalate(level, "high")
            flags.append("Stage 2-ish hypertension range (heuristic) → needs careful cardiovascular assessment.")
        elif p.sbp >= 130 or p.dbp >= 80:
            level = escalate(level, "med")
            flags.append("Elevated BP range (heuristic) → stimulants can increase BP/HR; monitor per standard practice.")

    if p.heart_rate is not None:
        if p.heart_rate >= 120:
            level = escalate(level, "high")
            flags.append("Tachycardia (HR ≥120 heuristic) → stimulant risk; reassess.")
        elif p.heart_rate >= 100:
            level = escalate(level, "med")
            flags.append("HR elevated (≥100 heuristic) → stimulant may increase HR; monitor.")

    # Renal note (some XR labeling includes ESRD not recommended)
    if p.egfr is not None and p.egfr < 15:
        level = escalate(level, "high")
        flags.append("eGFR very low (<15 heuristic) → some XR labeling notes not recommended in ESRD; verify current label.")

    # Insomnia
    if p.sleep_severe_insomnia == "Yes":
        level = escalate(level, "med")
        flags.append("Severe insomnia → stimulant timing/selection requires careful planning (label notes insomnia risk).")

    # Age
    if p.age is not None and p.age < 6:
        level = escalate(level, "high")
        flags.append("Age <6 selected → many amphetamine products have limited/age-specific labeling; verify indication/label.")

    return level, flags

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown(DISCLAIMER)

colA, colB = st.columns([1, 1])

with colA:
    med = st.selectbox(
        "Select medicine",
        list(LABEL_REFERENCE.keys()),
        index=0
    )
    indication = st.selectbox(
        "Select indication (demo)",
        list(LABEL_REFERENCE[med]["indications"].keys()),
        index=0
    )

with colB:
    st.info("Tip: You can leave most fields blank. The triage becomes more informative as more parameters are added.")

st.divider()
st.subheader("Patient parameters (optional)")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=None, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=400.0, value=None, step=0.5)
    pregnancy = st.selectbox("Pregnancy/Breastfeeding", ["No/Not applicable", "Pregnant", "Breastfeeding", "Unknown"])

with c2:
    sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=None, step=1)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=None, step=1)
    heart_rate = st.number_input("Heart rate (bpm)", min_value=0, max_value=250, value=None, step=1)

with c3:
    cardiac_disease = st.selectbox("Known symptomatic cardiovascular disease/serious heart disease?", ["No", "Yes", "Unknown"])
    glaucoma = st.selectbox("Glaucoma status", ["No", "Open-angle", "Acute narrow-angle", "Unknown"])
    hyperthyroidism = st.selectbox("Hyperthyroidism?", ["No", "Yes", "Unknown"])

st.markdown("#### Liver / kidney (optional labs)")
l1, l2, l3, l4 = st.columns(4)

with l1:
    liver_disease_known = st.selectbox("Known significant liver disease?", ["No", "Yes", "Unknown"])
with l2:
    alt = st.number_input("ALT (U/L)", min_value=0.0, max_value=5000.0, value=None, step=1.0)
with l3:
    ast = st.number_input("AST (U/L)", min_value=0.0, max_value=5000.0, value=None, step=1.0)
with l4:
    bilirubin = st.number_input("Total bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=None, step=0.1)

k1, k2, k3 = st.columns(3)
with k1:
    egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, max_value=200.0, value=None, step=1.0)
with k2:
    maoi_last_14d = st.selectbox("MAOI used in last 14 days?", ["No", "Yes", "Unknown"])
with k3:
    substance_use_disorder = st.selectbox("Substance use disorder history?", ["No", "Yes", "Unknown"])

m1, m2 = st.columns(2)
with m1:
    severe_anxiety_agitation = st.selectbox("Severe anxiety / agitation currently?", ["No", "Yes", "Unknown"])
with m2:
    sleep_severe_insomnia = st.selectbox("Severe insomnia currently?", ["No", "Yes", "Unknown"])

p = PatientInputs(
    age=age if age != 0 else age,
    weight_kg=weight_kg,
    pregnancy=pregnancy,
    sbp=sbp,
    dbp=dbp,
    heart_rate=heart_rate,
    cardiac_disease=cardiac_disease,
    glaucoma=glaucoma,
    hyperthyroidism=hyperthyroidism,
    liver_disease_known=liver_disease_known,
    alt=alt,
    ast=ast,
    bilirubin=bilirubin,
    egfr=egfr,
    maoi_last_14d=maoi_last_14d,
    substance_use_disorder=substance_use_disorder,
    severe_anxiety_agitation=severe_anxiety_agitation,
    sleep_severe_insomnia=sleep_severe_insomnia
)

st.divider()
st.subheader("Output")

left, right = st.columns([1, 1])

with left:
    st.markdown("### Label reference (non-personalized)")
    ref = LABEL_REFERENCE[med]["indications"][indication]["reference_dosing"]
    st.write("\n".join([f"- {x}" for x in ref]))

    contra = LABEL_REFERENCE[med]["indications"][indication].get("absolute_contra", [])
    if contra:
        st.markdown("**Key contraindications (label summary)**")
        st.write("\n".join([f"- {x}" for x in contra]))

    renal_note = LABEL_REFERENCE[med]["indications"][indication].get("renal_note", [])
    if renal_note:
        st.markdown("**Renal note**")
        st.write("\n".join([f"- {x}" for x in renal_note]))

with right:
    st.markdown("### Risk / caution triage")
    if med.startswith("Clonazepam"):
        level, flags = triage_clonazepam(p)
    else:
        level, flags = triage_adderall(p)

    # Pretty badge
    badge = {
        "low": "🟢 LOW",
        "med": "🟡 MED",
        "high": "🟠 HIGH",
        "severe": "🔴 SEVERE"
    }[level]
    st.markdown(f"## {badge}")

    if flags:
        st.markdown("**Triggered flags (what drove the rating):**")
        for f in flags:
            st.write(f"- {f}")
    else:
        st.success("No major risk flags triggered based on the fields provided.")

st.divider()
st.caption("Next step: if you want, we can plug this into your Dashboard 1 via multipage navigation (Streamlit pages) and keep a shared patient context.")
