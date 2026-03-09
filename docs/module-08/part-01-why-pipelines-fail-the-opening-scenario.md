## Part 1: Why pipelines fail -- the opening scenario

Before writing a line of code, you need to understand what you are building against.

### The NCB encoding incident

A UK private motor book runs a model refresh. The modelling team fits a CatBoost Poisson frequency model. The Gini lifts from 0.33 to 0.41. The out-of-time validation on accident year 2024 is clean. The SHAP relativities show NCB as the dominant factor, with a sensible monotone pattern: NCD 0 is 2.3x the base, NCD 5 is 0.62x. The pricing committee approves deployment. The model goes live in March 2025.

In June 2025 the actual-versus-expected ratio by NCD band stops making sense. NCD 0 policies are running significantly better than the model predicted. NCD 5 policies are running significantly worse. The pricing actuary pulls the scoring pipeline code.

The training pipeline encodes NCB years as a string: `pl.col("ncb_years").cast(pl.Utf8)`. CatBoost treats it as a categorical. The scoring pipeline, written by a different team member six weeks later from memory, passes `ncb_years` as an integer. CatBoost treats it as continuous. The model has been live for three months applying systematically wrong feature values. No exception was raised. The predictions looked plausible.

The monetary cost: one full quarter of mispriced renewals. The regulatory cost: a Consumer Duty breach, because the pricing was not what the model was validated to produce.

The fix is not model improvement. The fix is a shared feature engineering layer that both the training pipeline and the scoring pipeline import. One function. One encoding. No duplication.

### The development data contamination incident

A London specialty insurer builds a severity model for solicitors' professional indemnity. The validation metrics look reasonable. Six months after deployment, the loss ratio is running 18 points above the model's prediction. An investigation finds that the training data included IBNR-contaminated claims: the most recent 24 months of claims in the training set had not yet developed to ultimate. The model learned to predict incurred-to-date severity, not ultimate severity. A model trained on partially developed claims will always produce optimistic severity predictions for similar risks in the future.

The fix is an IBNR buffer in the cross-validation structure. The buffer removes the most recent N months from the end of each training fold. For motor third-party property, six months is sufficient. For solicitors' PI, 24 months is the minimum.

### The version confusion incident

A mid-size regional insurer runs a model review in September 2025. Three months later, the FCA requests documentation of the pricing basis for a specific cohort of policies. The pricing actuary cannot identify which model version was deployed, what data it was trained on, or what validation metrics it achieved. The answer turns out to be "the pricing actuary's laptop, which has since been refreshed."

The fix is an audit record. Every pipeline run writes one row to a Delta table: model run IDs, data table versions, validation metrics, run date, run configuration. Delta's time travel preserves historical data table versions. MLflow preserves model artefacts. The audit record ties them together.

### What these three incidents have in common

None of them were modelling failures. The models were technically sound. The failures were in the connections between components -- between training and scoring, between training and validation, between production and audit. An end-to-end pipeline disciplines those connections. That is what this module builds.