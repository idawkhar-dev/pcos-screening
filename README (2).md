# PCOS Screening Calculator

An educational & awareness screening tool for PCOS. It takes eight questionnaire answers and returns a
risk level: Low / Moderate / High and the factors that contributed most.

**Not a diagnostic tool.** Outputs are for screening, prioritisation and education only.

**Live demo/Use Calculator using this link:** https://riskcalculator-pcos.streamlit.app/

---

## Data

| | |
|---|---|
| Source | [Polycystic Ovary Syndrome (PCOS)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) — Prasoon Kottarathil, Kaggle, 2020 |
| Data Distribution | 541 women evaluated for suspected PCOS across 10 hospitals in Kerala, India. 364 reported negative for PCOS and 177 reported positive. 

## Model

Logistic regression (`impute → standardise → logistic`), trained on **questionnaire-answerable
features only**. The model is not trained on lab or ultrasound inputs and the clinical data features of the data. Since the clinical features are harder for users to enter, they are not considered.

**Features used (8):** irregular cycles, excess hair growth, skin darkening, weight gain, acne,
fast food intake, age, period length.

Weights, level cut-offs and display labels are exported to `pcos_model.json`; the app scores in
numpy, so scikit-learn is not needed at runtime.

## Results

Five-fold cross-validated ROC AUC:

| Feature set | AUC |
|---|---|
| All features (41) clinical + questions data | 0.944 |
| All questions data features (11) | 0.883 |
| **Final model (top 8 features)** | **0.886** |

Follicle counts are the strongest correlates of the label (r = 0.65 / 0.60) because they *are* the
ultrasound diagnostic criterion but are excluded deliberately.

**Risk Levels** (probability cut-offs 0.25 / 0.55), and the observed PCOS rate in each Level:

| Level |PCOS rate |
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
app.py                      Streamlit questions + scoring app backend
pcos_model.json             model weights, level cut-offs, display labels
EAD_MODEL_PCOS.ipynb        EDA, feature selection, and model training
test_validation.py          input- output test cases 
requirements.txt
```

Full problem framing, data limitations and next steps are covered in the following
presentation: 
