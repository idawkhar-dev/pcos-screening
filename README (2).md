# PCOS Screening Calculator

An educational screening tool for PCOS (PMOS). It takes eight questionnaire answers and returns a
risk band — Low / Moderate / High — plus the factors that contributed most.

**Not a diagnostic tool.** Outputs are for screening, prioritisation and education only.

**Live demo:** <add-your-streamlit-url>

---

## Data

| | |
|---|---|
| Source | [Polycystic Ovary Syndrome (PCOS)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) — Prasoon Kottarathil, Kaggle, 2020 |
| Cohort | 541 women evaluated for suspected PCOS across 10 hospitals in Kerala, India |
| Class balance | 364 negative / 177 positive (32.7% positive) |
| Licence | Public Kaggle dataset; no PHI redistributed |

## Model

Logistic regression (`impute → standardise → logistic`), trained on **questionnaire-answerable
features only**. No lab or ultrasound inputs.

**Features used (8):** irregular cycles, excess hair growth, skin darkening, weight gain, acne,
fast food intake, age, period length.

Weights, band cut-offs and display labels are exported to `pcos_model.json`; the app scores in
numpy, so scikit-learn is not needed at runtime.

## Results

Five-fold cross-validated ROC AUC:

| Feature set | AUC |
|---|---|
| All features (41) | 0.944 |
| Clinical / ultrasound only (11) | 0.901 |
| All questionnaire features (11) | 0.883 |
| **Final model — top 8 questionnaire features** | **0.886** |

Follicle counts are the strongest correlates of the label (r = 0.65 / 0.60) because they *are* the
ultrasound diagnostic criterion — they are excluded deliberately.

**Risk bands** (probability cut-offs 0.25 / 0.55), and the observed PCOS rate in each band:

| Band | Cohort PCOS rate |
|---|---|
| Low | 0.09 |
| Moderate | 0.28 |
| High | 0.83 |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
app.py                      Streamlit questionnaire + scoring
pcos_model.json             model weights, band cut-offs, display labels
EAD_MODEL_PCOS.ipynb        EDA, feature selection, training, export
test_validation.py          sanity tests + input-space band sweep
requirements.txt
```

Full problem framing, data limitations and next steps are covered in the accompanying
presentation.
