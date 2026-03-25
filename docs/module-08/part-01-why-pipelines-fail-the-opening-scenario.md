## Part 1: Why pipelines fail — the opening scenario

Three incidents. None of them were modelling failures.

### The NCB encoding incident

A UK private motor book runs a model refresh in early 2025. The modelling team fits a CatBoost Poisson frequency model on two years of policy data. The Gini lifts from 0.33 to 0.41. Out-of-time validation on accident year 2024 looks clean. The SHAP relativities show NCB as the dominant factor with the expected monotone pattern: NCD 0 at 2.3x the base rate, NCD 5 at 0.62x. The pricing committee approves deployment.

In June 2025 the actual-versus-expected ratio by NCD band stops making sense. NCD 0 policies are running better than the model predicted. NCD 5 policies are running worse. The pricing actuary pulls the scoring pipeline code.

The training pipeline encodes NCB years as a string: `pl.col("ncb_years").cast(pl.Utf8)`. CatBoost treats it as categorical. The scoring pipeline, written from memory six weeks later, passes `ncb_years` as an integer. CatBoost treats it as continuous. The model has been live for three months applying systematically wrong relativities. No exception was raised. The predictions looked plausible because NCD 0 and NCD 5 are at opposite ends of both the categorical and numeric ranges — they point in the same direction, just at different magnitudes.

**The fix is not model improvement.** The fix is a shared feature engineering layer: one function, one encoding, imported by both the training notebook and the scoring code. This module implements that pattern in Stage 3.

### The IBNR contamination incident

A London specialty insurer builds a severity model for solicitors' professional indemnity in Q4 2024. Validation metrics look reasonable. Six months after deployment, the loss ratio is running 18 points above the model's prediction.

The investigation finds that the training data included the most recent 24 months of claims at face value. Solicitors' PI claims can take 5-7 years to develop to ultimate. The model learned to predict incurred-to-date severity on 18-month-old claims, not ultimate severity. A model trained on partially developed claims will always produce optimistic severity predictions — the training target is structurally lower than the deployment target.

**The fix is a development buffer in the cross-validation structure.** Remove the most recent N months from each training fold, forcing the model to learn from claims that have had time to develop. For motor third-party property damage, 6 months is usually sufficient. For motor bodily injury, 12-18 months. For solicitors' PI or employers' liability, 24 months is the minimum. This module implements the buffer in Stage 4.

### The version confusion incident

A mid-size regional insurer runs a model review in September 2025. In December 2025, the FCA requests documentation of the pricing basis for a specific cohort of policies. The pricing actuary cannot identify which model version was deployed for that cohort, what data it was trained on, or what validation metrics it achieved at sign-off.

After investigation: the model was trained in the pricing actuary's local Python environment, which has since been rebuilt. The training data was an Excel extract from the claims system, now overwritten. The validation metrics were in an email that predates the firm's email archiving policy.

**The fix is an audit record.** Every pipeline run writes one append-only row to a Delta table: model run IDs, data table versions, validation metrics, run configuration, run date. Delta's time travel preserves every historical version of the data tables. MLflow preserves the model artefacts. The audit record ties them together with a key that lets you reconstruct any pipeline run from any point in the past. Stage 10 of this module builds that record.

### What these three incidents have in common

All three involved technically sound models. The models were not the problem. The connections between components were the problem — between training and scoring, between training data and validation, between production output and the audit trail.

A pipeline disciplines those connections. That is what this module builds.

---

### The organisational case for pipelines

The technical case is straightforward: shared transforms prevent encoding divergence, temporal splits prevent data leakage, audit records satisfy regulatory requests. The organisational case is less often articulated but equally important.

A pricing review that runs from raw data to rate action in a single reproducible notebook can be handed off. When the pricing actuary who built it leaves, the next person can re-run it, understand its structure, and modify it without reconstructing it from scattered notebooks and emails. When the underwriting director asks "what does the model say about London BI risk specifically?", the answer is a SQL query on the freq_predictions Delta table, not a manual re-run. When the FCA asks for the methodology, the answer is the notebook, the MLflow run IDs, and the audit table — not a written reconstruction from memory.

The pipeline is organisational infrastructure as much as technical infrastructure. Build it once. Run it quarterly. Every review thereafter is a re-run with updated data, not a rebuild from scratch.
