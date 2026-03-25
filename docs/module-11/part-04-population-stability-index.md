## Part 4: Population Stability Index (PSI)

### What PSI measures

PSI measures whether a distribution has shifted between two time periods. It was originally developed in credit scoring to detect when the score distribution on new applicants had drifted from the score distribution on which the scorecard was validated. The same concept applies to any numeric distribution: feature values, predicted scores, or anything else with a distribution.

PSI is calculated by binning the reference distribution into N buckets, then comparing the proportion of the current distribution that falls in each bucket:

```text
PSI = sum( (current_pct - reference_pct) × ln(current_pct / reference_pct) )
```

The thresholds are:

| PSI | Interpretation |
|-----|----------------|
| < 0.10 | No significant change |
| 0.10 - 0.25 | Moderate change, worth monitoring |
| > 0.25 | Significant change, investigate |

These thresholds come from credit scoring practice — they are not derived from any rigorous statistical theory. They are empirical rules of thumb that have proven useful in practice. We use them because they are the industry standard, not because they are optimal.

### Why we run PSI on the score distribution first

Before we look at individual features, we run PSI on the predicted score distribution. If PSI on the scores is low (< 0.10), whatever has changed about the input data has not materially changed the risk distribution. That is a mild signal at most.

If PSI on the scores is high (> 0.25), something has changed the risk distribution significantly. The CSI analysis then tells us which features are driving that change.

### Computing predicted scores

First, load the trained model from MLflow. We trained and registered this in Module 8:

```python
import mlflow.catboost

model = mlflow.catboost.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)
print("Model loaded OK")
```

Now generate predictions for both windows:

```python
from catboost import Pool
import numpy as np

feature_names = model.feature_names_

ref_pool = Pool(df_reference.select(feature_names).to_pandas(), cat_features=["region"])
cur_pool = Pool(df_current.select(feature_names).to_pandas(), cat_features=["region"])

pred_ref = model.predict(ref_pool)
pred_cur = model.predict(cur_pool)

exposure_ref = df_reference["exposure"].to_numpy()
exposure_cur = df_current["exposure"].to_numpy()
```

### Running PSI

```python
from insurance_monitoring.drift import psi

psi_score = psi(
    reference=pred_ref,
    current=pred_cur,
    n_bins=N_BINS,
    exposure_weights=exposure_cur,    # weight by exposure (correct for insurance)
    reference_exposure=exposure_ref,
)

print(f"PSI (score distribution): {psi_score:.4f}")
if psi_score < 0.10:
    print("Green: no significant change")
elif psi_score < 0.25:
    print("Amber: moderate change, investigate")
else:
    print("Red: significant change, investigate")
```

`psi()` returns a single float — the Population Stability Index. The `exposure_weights` and `reference_exposure` arguments tell the function to weight each observation by its exposure when computing the bucket proportions. This is the correct approach for insurance data: a policy with 0.25 years of exposure should not count the same as a policy with a full year when computing the score distribution.

Without exposure weighting, a December spike in new short-term policies would distort the current distribution towards the short-term risk profile, and PSI would flag a shift that is just a seasonal artefact.

### Visualising the distribution shift

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pred_ref, bins=50, alpha=0.6, label="Reference", color="steelblue", density=True)
ax.hist(pred_cur, bins=50, alpha=0.6, label="Current",   color="tomato",    density=True)
ax.axvline(np.median(pred_ref), color="steelblue", linestyle="--", alpha=0.8, label="Ref median")
ax.axvline(np.median(pred_cur), color="tomato",    linestyle="--", alpha=0.8, label="Cur median")
ax.set_xlabel("Predicted frequency")
ax.set_ylabel("Density")
ax.set_title(f"Score distribution  (PSI = {psi_score:.3f})")
ax.legend()
plt.tight_layout()
plt.show()
```

### Interpreting the result

A PSI below 0.10 means the score distribution has not changed materially. Proceed to CSI to check individual features as a routine step.

A PSI between 0.10 and 0.25 means the score distribution has shifted somewhat. Check which features are driving this via CSI. No immediate action on the model, but note it in the monitoring log.

A PSI above 0.25 means the score distribution has shifted significantly. This is an amber or red flag depending on the A/E ratio. If the A/E is also elevated, this is a calibration problem. If the A/E is within tolerance, the model is re-ordering risks differently but may still be correctly calibrated overall.

Store the PSI result:

```python
# Store the scalar PSI score for later comparison.
# Note: MonitoringReport.score_reference and score_current expect full score
# arrays (not this scalar) — the report computes PSI internally from those arrays.
# See Part 8 for the correct MonitoringReport construction.
print(f"PSI score stored: {psi_score:.4f}")
```
