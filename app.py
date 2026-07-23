import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PCOS Screening", page_icon="🌸", layout="centered")


# ---------------------------------------------------------------- model ----
def load_model(path="pcos_model.json"):
    with open(path) as f:
        return json.load(f)


M = load_model()
FEATURES, LABELS = M["features"], M["labels"]
LOW_CUT, HIGH_CUT = M["low_cut"], M["high_cut"]


def risk_band(p):
    return "Low" if p < LOW_CUT else ("Moderate" if p < HIGH_CUT else "High")


def predict_risk(answers: dict):
    """Reimplements the sklearn pipeline (impute -> scale -> logistic) in numpy."""
    x = np.array([answers.get(f, np.nan) for f in FEATURES], dtype=float)
    x = np.where(np.isnan(x), np.array(M["medians"]), x)        # impute
    z = (x - np.array(M["means"])) / np.array(M["scales"])      # standardise
    w = np.array(M["coefficients"])
    p = 1 / (1 + np.exp(-(M["intercept"] + float(z @ w))))      # sigmoid

    # per-feature contribution drives the "why" shown to the user
    contrib = pd.Series(z * w, index=FEATURES)
    top = [LABELS[f] for f in contrib[contrib > 0].sort_values(ascending=False).head(3).index]
    band = risk_band(p)
    return p, band, ([] if band == "Low" else top)


# ------------------------------------------------------------ questions ----
QUESTIONS = [
    {"key": "Cycle(R/I)",
     "q": "How would you describe your menstrual cycles?",
     "help": "Irregular means unpredictable timing, or fewer than about nine periods a year.",
     "type": "choice", "options": [("Regular", 2), ("Irregular", 4)]},

    {"key": "hair growth(Y/N)",
     "q": "Do you have excess hair growth on your face or body?",
     "help": "For example on the chin, upper lip, chest or back.",
     "type": "choice", "options": [("No", 0), ("Yes", 1)]},

    {"key": "Skin darkening (Y/N)",
     "q": "Have you noticed darkened patches of skin?",
     "help": "Often on the back of the neck, underarms or groin.",
     "type": "choice", "options": [("No", 0), ("Yes", 1)]},

    {"key": "Weight gain(Y/N)",
     "q": "Have you had unexplained weight gain recently?",
     "help": "Weight gain without a clear change in diet or activity.",
     "type": "choice", "options": [("No", 0), ("Yes", 1)]},

    {"key": "Pimples(Y/N)",
     "q": "Do you get frequent acne or pimples?",
     "help": "Persistent acne, especially along the jaw or chin.",
     "type": "choice", "options": [("No", 0), ("Yes", 1)]},

    {"key": "Fast food (Y/N)",
     "q": "Do you eat fast food frequently?",
     "help": "Roughly several times a week or more.",
     "type": "choice", "options": [("No", 0), ("Yes", 1)]},

    {"key": "Age (yrs)",
     "q": "How old are you?",
     "help": "",
     "type": "number", "min": 15, "max": 48, "default": 27, "unit": "years"},

    {"key": "Cycle length(days)",
     "q": "How many days does your period usually last?",
     "help": "The number of days of bleeding, not the gap between periods.",
     "type": "number", "min": 2, "max": 12, "default": 5, "unit": "days"},
]

DISCLAIMER = (
    "**This is an educational screening tool, not a diagnosis.** It cannot tell you whether you "
    "have PCOS. Only a clinician can diagnose PCOS, using clinical criteria, examination and "
    "tests. If you have concerns about your health, please speak to a healthcare professional."
)


# ---------------------------------------------------------------- state ----
if "stage" not in st.session_state:
    st.session_state.stage = "intro"
    st.session_state.idx = 0
    st.session_state.answers = {}


def restart():
    st.session_state.stage = "intro"
    st.session_state.idx = 0
    st.session_state.answers = {}


def answer(key, value):
    st.session_state.answers[key] = value
    if st.session_state.idx + 1 >= len(QUESTIONS):
        st.session_state.stage = "result"
    else:
        st.session_state.idx += 1


# ------------------------------------------------------------------ css ----
st.markdown("""
<style>
/* page width + breathing room */
.block-container {max-width: 640px; padding-top: 2.5rem; padding-bottom: 3rem;}

/* cards */
.card {background:#fff; border:1px solid #ece3f2; border-radius:18px;
       padding:32px 30px; box-shadow:0 3px 14px rgba(80,40,100,.06);
       margin-bottom:26px; text-align:center;}
.card h3 {margin:10px 0 10px 0; font-size:1.35rem; line-height:1.45; font-weight:600;}
.card p  {margin:0; color:#7a7280; font-size:.92rem; line-height:1.55;}
.qnum {color:#a06fb5; font-size:.72rem; letter-spacing:.12em;
       text-transform:uppercase; font-weight:600;}

/* result card */
.result-band {font-size:2.1rem; font-weight:700; margin:14px 0 12px 0;}
.result-card {padding:40px 32px;}

/* --- buttons: force full width of their column, never shrink-wrap --- */
.stButton {width:100% !important;}
.stButton > button,
button[data-testid^="stBaseButton"] {
    width:100% !important;
    min-height:52px !important;
    padding:0.75rem 1.25rem !important;
    border-radius:12px !important;
    font-weight:500 !important;
    font-size:1rem !important;
    white-space:nowrap !important;
    border:1.5px solid #ddd0e6 !important;
    background:#fff !important;
    color:#4a3d52 !important;
    transition:all .15s ease !important;
}
.stButton > button:hover,
button[data-testid^="stBaseButton"]:hover {
    border-color:#a06fb5 !important;
    color:#7b4f92 !important;
    background:#faf6fc !important;
}
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background:#8e5aa8 !important;
    border-color:#8e5aa8 !important;
    color:#fff !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background:#7b4f92 !important;
    border-color:#7b4f92 !important;
}

/* softer progress bar */
.stProgress > div > div > div > div {background-color:#c9a5d8;}

/* contributing-factor chips */
.factor {background:#faf6fc; border-left:3px solid #c9a5d8; border-radius:6px;
         padding:10px 14px; margin-bottom:8px; font-size:.95rem; text-align:left;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- intro ----
if st.session_state.stage == "intro":
    st.markdown("<div style='text-align:center'><h1>🌸 PCOS Risk Calculator</h1></div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='card'><p>This calculator records  response to eight quick questions about symptoms commonly associated with "
        "PCOS (polycystic ovary syndrome). At the end of this form, you will see whether your answers flag "
        "high, medium or low likelihood of being dignosed with PCOS and which factors matter the most for the flagged risk.</p></div>",
        unsafe_allow_html=True)
    st.warning(DISCLAIMER)
    st.caption("Takes about a minute. Nothing you enter is stored.")
    st.write("")
    c = st.columns([1, 2, 1])
    with c[1]:
        if st.button("Start the quiz", type="primary", use_container_width=True):
            st.session_state.stage = "quiz"
            st.rerun()


# ----------------------------------------------------------------- quiz ----
elif st.session_state.stage == "quiz":
    i = st.session_state.idx
    q = QUESTIONS[i]

    st.progress(i / len(QUESTIONS))
    st.markdown(
        f"<div class='card'><div class='qnum'>Question {i + 1} of {len(QUESTIONS)}</div>"
        f"<h3>{q['q']}</h3><p>{q['help']}</p></div>",
        unsafe_allow_html=True)

    if q["type"] == "choice":
        left, right = st.columns(2, gap="medium")
        for col, (label, val) in zip((left, right), q["options"]):
            with col:
                if st.button(label, key=f"{q['key']}_{val}", use_container_width=True):
                    answer(q["key"], val)
                    st.rerun()
    else:
        val = st.slider(q["unit"], q["min"], q["max"], q["default"], key=f"sl_{q['key']}")
        st.write("")
        c = st.columns([1, 2, 1])
        with c[1]:
            if st.button("Continue", type="primary", use_container_width=True):
                answer(q["key"], val)
                st.rerun()

    if i > 0:
        st.write("")
        c = st.columns([1, 2, 1])
        with c[1]:
            if st.button("← Back", use_container_width=True):
                st.session_state.idx -= 1
                st.rerun()


# --------------------------------------------------------------- result ----
elif st.session_state.stage == "result":
    p, band, top = predict_risk(st.session_state.answers)
    color = {"Low": "#2e7d32", "Moderate": "#ef6c00", "High": "#c62828"}[band]
    blurb = {
        "Low": "Your answers suggest a lower likelihood of PCOS. Keep tracking your cycle, and "
               "see a clinician if your symptoms change.",
        "Moderate": "Some of your answers are associated with PCOS. It would be worth raising "
                    "with a clinician at your next visit.",
        "High": "Several of your answers are commonly associated with PCOS. Consider booking a "
                "clinical evaluation to look into it.",
    }[band]

    st.markdown(
        f"<div class='card result-card'><div class='qnum'>Your result</div>"
        f"<div class='result-band' style='color:{color}'>{band} likelihood</div>"
        f"<p>{blurb}</p></div>",
        unsafe_allow_html=True)

    if top:
        st.markdown("<p style='text-align:center; font-weight:600; margin-bottom:12px'>"
                    "What contributed most to this result</p>", unsafe_allow_html=True)
        for t in top:
            st.markdown(f"<div class='factor'>{t}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center; color:#7a7280'>"
                    "Nothing in your answers pointed strongly toward PCOS.</p>",
                    unsafe_allow_html=True)

    st.write("")
    st.info(DISCLAIMER)

    with st.expander("How this works"):
            st.markdown(
            "**Where the result comes from:** This tool looks at a public clinical dataset of "
            "541 women who were evaluated for suspected PCOS (Kaggle, 2020; 10 hospitals in "
            "Kerala, India). It learned which *combinations* of symptoms showed up most often "
            "in the women who turned out to have PCOS, and how much each symptom mattered "
            "relative to the others. Your answers are compared against those patterns. The "
            "result reflects how closely your combination of symptoms resembles theirs, not a clinical "
            "test of your body.\n\n"
            "Users are encouraged to treat this as a nudge to start a "
            "conversation, not a clinical diagnosis of their health."
        )

    st.write("")
    c = st.columns([1, 2, 1])
    with c[1]:
        if st.button("Start over", use_container_width=True):
            restart()
            st.rerun()