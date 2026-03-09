## Part 4: Population Stability Index (PSI)

### What PSI measures

PSI measures whether a distribution has shifted between two time periods. It was originally developed in credit scoring to detect when the score distribution on new applicants had drifted from the score distribution on which the scorecard was validated. The same concept applies to any numeric distribution: feature values, predicted scores, or anything else with a distribution.

PSI is calculated by binning the reference distribution into N buckets, then comparing the proportion of the current distribution that falls in each bucket:

```text
PSI = sum( (current_pct - reference_pct) * ln(current_pct / reference_pct) )
```

The thresholds are:

| PSI | Interpretation |
|-----|----------------|
| < 0.10 | No significant change |
| 0.10 - 0.20 | Minor change, worth monitoring |
| > 0.20 | Significant change, investigate |

These thresholds come from credit scoring practice - they are not derived from any rigorous statistical theory. They are empirical rules of thumb that have proven useful in practice. We use them because they are the industry standard, not because they are optimal.

### Why we run PSI on the score distribution first

Before we look at individual features, we run PSI on the predicted score distribution. If PSI on the scores is low (< 0.10), whatever has changed about the input data has not materially changed the risk distribution. That is a mild signal at most.

If PSI on the scores is high (> 0.20), something has changed the risk distribution significantly. The CSI analysis then tells us which features are driving that change.

### Computing predicted scores

First, load the trained model from MLflow. We trained and registered this in Module 8:

```python
import mlflow.catboost

# Load the registered model
model = mlflow.catboost.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)
print("Model loaded OK")
print(f"  Type: {type(model)}")
```

If the model is not in the registry, load it from a local path:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("/dbfs/models/motor_frequency_catboost.cbm")
```

Now generate predictions for both windows. We need to pass features in the same order the model was trained on. The feature names are stored in the model:

```python
# Get feature names from the model
feature_names = model.feature_names_

print(f"Model expects {len(feature_names)} features:")
for f in feature_names:
    print(f"  {f}")
```

Generate predictions:

```python
# Convert to pandas for CatBoost (bridge between Polars and CatBoost)
X_ref = df_reference.select(feature_names).to_pandas()
X_cur = df_current.select(feature_names).to_pandas()

# Predict - these are predicted claim frequencies (annualised)
pred_ref = model.predict(X_ref)
pred_cur = model.predict(X_cur)

# Scale by exposure to get predicted claim counts
exposure_ref = df_reference["exposure"].to_numpy()
exposure_cur = df_current["exposure"].to_numpy()

pred_count_ref = pred_ref * exposure_ref
pred_count_cur = pred_cur * exposure_cur

print(f"Reference predictions: mean={pred_ref.mean():.4f}, "
      f"min={pred_ref.min():.4f}, max={pred_ref.max():.4f}")
print(f"Current predictions:   mean={pred_cur.mean():.4f}, "
      f"min={pred_cur.min():.4f}, max={pred_cur.max():.4f}")
```

The predictions are predicted annualised frequencies. We run PSI on the raw frequency predictions (not the count predictions), because frequency is what the model is scoring on.

### Running PSI

```python
from insurance_monitoring import PSICalculator

psi_calc = PSICalculator(n_bins=N_BINS)

psi_result = psi_calc.calculate(
    reference=pred_ref,
    current=pred_cur,
    exposure_ref=exposure_ref,   # optional but recommended
    exposure_cur=exposure_cur,
)

print(f"PSI (score distribution): {psi_result.psi:.4f}")
print(f"Traffic light: {psi_result.traffic_light}")
print()
print("Bin-level breakdown:")
for bin_info in psi_result.bins:
    print(f"  [{bin_info.lower:.3f}, {bin_info.upper:.3f}): "
          f"ref={bin_info.ref_pct:.2%}, "
          f"cur={bin_info.cur_pct:.2%}, "
          f"contribution={bin_info.contribution:.4f}")
```

The `exposure_ref` and `exposure_cur` arguments tell the calculator to weight each observation by its exposure when computing the bucket percentages. This is the correct approach for insurance data - a policy with 0.25 years of exposure should not count the same as a policy with a full year when computing the score distribution.

Without exposure weighting, a December spike in new short-term policies would distort the current distribution towards the short-term risk profile, and PSI would flag a shift that is just a seasonal artefact.

### Visualising the PSI breakdown

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Score distributions
ax1.hist(pred_ref, bins=50, alpha=0.6, label="Reference", color="steelblue", density=True)
ax1.hist(pred_cur, bins=50, alpha=0.6, label="Current", color="tomato", density=True)
ax1.set_xlabel("Predicted frequency")
ax1.set_ylabel("Density")
ax1.set_title(f"Score distribution  (PSI = {psi_result.psi:.3f})")
ax1.legend()
ax1.axvline(np.median(pred_ref), color="steelblue", linestyle="--", alpha=0.8, label="Ref median")
ax1.axvline(np.median(pred_cur), color="tomato", linestyle="--", alpha=0.8, label="Cur median")

# Bin-level contributions
bins_lower = [b.lower for b in psi_result.bins]
contributions = [b.contribution for b in psi_result.bins]
colors = ["green" if c < 0.02 else "orange" if c < 0.05 else "red" for c in contributions]
ax2.bar(range(len(contributions)), contributions, color=colors)
ax2.set_xlabel("PSI bin")
ax2.set_ylabel("Contribution to PSI")
ax2.set_title("PSI contribution by bin")
ax2.axhline(0.02, color="orange", linestyle="--", alpha=0.7, label="Amber threshold")
ax2.axhline(0.05, color="red", linestyle="--", alpha=0.7, label="Red threshold")
ax2.legend()

plt.tight_layout()
plt.savefig("/tmp/psi_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /tmp/psi_score_distribution.png")
```

Look at the per-bin contributions. If two or three bins are driving most of the PSI, the shift is concentrated at specific risk levels. A shift concentrated in the high-risk bins (right tail) is more concerning than a diffuse shift across all bins, because it means your most expensive risks are behaving differently.

### Interpreting the result

A PSI below 0.10 means the score distribution has not changed materially. Proceed to CSI to check individual features as a routine step.

A PSI between 0.10 and 0.20 means the score distribution has shifted somewhat. Check which features are driving this via CSI. No immediate action on the model, but note it in the monitoring log.

A PSI above 0.20 means the score distribution has shifted significantly. This is an amber or red flag depending on the A/E ratio. If the A/E is also elevated, this is a calibration problem. If the A/E is within tolerance, the model is re-ordering risks differently but may still be correctly calibrated overall - a nuanced situation that requires further investigation.

We cover the combined interpretation in Part 9. For now, store the PSI result:

```python
# Store for use in MonitoringReport (Part 8)
psi_score = psi_result
```
