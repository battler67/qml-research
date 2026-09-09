# Data dictionary and validation

## UCI Cleveland

| Field | Interpretation | Smoke representation |
| --- | --- | --- |
| age | age in years | continuous |
| sex | recorded sex code | binary code |
| cp | chest-pain category | categorical code |
| trestbps | resting blood pressure, mm Hg | continuous |
| chol | serum cholesterol, mg/dL | continuous |
| fbs | fasting blood sugar indicator | binary code |
| restecg | resting ECG result | categorical code |
| thalach | maximum heart rate achieved | continuous |
| exang | exercise-induced angina | binary code |
| oldpeak | exercise ST depression | continuous |
| slope | peak exercise ST slope | categorical code |
| ca | major vessels visualized | categorical/count code |
| thal | thallium-test category | categorical code |
| num | diagnostic outcome 0-4 | outcome only; never input |

`?` is parsed as missing. The smoke reduced comparison preserves the 13 original clinical fields as
numeric/coded fields so selection remains interpretable. Median imputation and scaling are fitted on
training data only. This compact categorical representation is a resource-bounded ablation; a full
practical classical benchmark should one-hot nominal fields. The original 0-4 severity is retained
for audit but smoke acceptance uses `num > 0` only.

## Framingham teaching data

Required cohort/outcome fields are `RANDID`, `PERIOD`, `PREVCHD`, `ANYCHD`, and `TIMECHD`.
Candidate baseline predictors include age, sex, systolic/diastolic pressure, total cholesterol, BMI,
smoking, cigarettes/day, diabetes, blood-pressure medication, heart rate, and glucose. Outcome and
future-time fields are programmatically forbidden as predictors. Negative sentinels and missing
markers must be converted according to the official documentation before fitting.

For the binary ten-year endpoint, a positive has `ANYCHD == 1` and `TIMECHD <= 3652.5`. A negative
has no event inside the horizon and observed follow-up through the horizon. Participants without an
event and with shorter follow-up are censored/unknown and excluded rather than relabelled negative.
