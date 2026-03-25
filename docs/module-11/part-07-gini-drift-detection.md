## Part 7: Gini drift detection

### What Gini drift measures

The Gini coefficient (equivalently, 2 × AUROC − 1 for binary outcomes) measures how well the model *ranks* risks. A Gini of 0 means the model has no discriminatory power. A Gini of 1 means the model ranks risks perfectly. A good UK motor frequency model achieves a Gini of around 0.35–0.50 on held-out data.

Gini drift is the question: has the Gini changed between the reference period and the current period? A falling Gini means the model is less able to distinguish high-risk from low-risk policies. This is a form of concept drift — the features that previously predicted claims well are no longer predicting them as well.

### Why Gini drift is a different signal from A/E

The A/E ratio tests calibration: is the model's average prediction correct? Gini tests discrimination: does the model correctly rank risks? These can move independently.

A model can have a perfect A/E of 1.0 but falling Gini. This happens when the model's predictions are re-scaled correctly (so the average is right) but the ranking has weakened (so the ordering is less accurate). This is dangerous for pricing: the average premium may be correct, but the distribution of premiums across the portfolio may be wrong. High-risk policies may be under-priced; low-risk policies may be over-priced.

Conversely, a model can have a good Gini but a shifted A/E. The model still ranks risks correctly but at the wrong level. This is a calibration problem: apply a multiplicative adjustment to fix the average, and the discrimination is fine.

Understanding which type of drift you have determines the response:

- Calibration problem (A/E off, Gini OK): apply a recalibration factor
- Discrimination problem (Gini dropping, A/E may or may not be affected): retraining required

### Computing Gini drift

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(actual_ref, pred_ref, exposure=exposure_ref)
gini_cur = gini_coefficient(actual_cur, pred_cur, exposure=exposure_cur)

# gini_drift_test returns a GiniDriftResult dataclass with fields:
#   reference_gini, current_gini, gini_change, z_statistic, p_value, significant
result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    reference_exposure=exposure_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    current_exposure=exposure_cur,
    n_bootstrap=200,
)

print(f"Gini (reference): {result.reference_gini:.4f}")
print(f"Gini (current):   {result.current_gini:.4f}")
print(f"Change:           {result.gini_change:+.4f}")
print(f"Z-statistic:      {result.z_statistic:.4f}")
print(f"P-value:          {result.p_value:.4f}")
print(f"Significant:      {result.significant}")
```

The test uses a bootstrap variance estimator (Algorithm 2 from arXiv 2510.04556) on both reference and current periods. The default significance level is alpha=0.32 (the "one-sigma rule" recommended by the paper for monitoring, which catches drift earlier than alpha=0.05 at the cost of more false positives). Use `alpha=0.05` for confirmatory testing.

`gini_change` is `current_gini - reference_gini`: a negative value means discrimination has declined. `significant` is True when `p_value < alpha`.

A p-value below the threshold means the Gini difference is statistically significant — the model's discrimination has changed, and it is unlikely to be due to random variation.

### Interpreting the result

A statistically significant fall in Gini is serious. If a model's Gini falls from 0.42 to 0.35, the model is less able to separate claimants from non-claimants. Risks that were in the top quintile of predicted frequency are now less likely to actually be high-frequency claims. The pricing order is becoming noisier.

Check whether the Gini fall is:

1. **Concentrated in a segment** — run `gini_coefficient()` separately for different segments (age bands, regions) to identify where discrimination has fallen most
2. **Consistent across time** — if you have monthly data, compute Gini in rolling windows to see if it is trending down or was a one-off
3. **Correlated with a CSI-flagged feature** — if vehicle group has drifted significantly and Gini has fallen, a likely explanation is that vehicle group is a key discriminator and its relationship to risk has changed

```python
# Segment Gini analysis — run for age bands
print("\nGini by driver age band:")
print(f"{'Band':<15} {'Gini ref':>10}  {'Gini cur':>10}  {'p-value':>10}")
print("-" * 50)

age_bands = [(17, 25, "17-24"), (25, 40, "25-39"), (40, 60, "40-59"), (60, 100, "60+")]

for low, high, label in age_bands:
    ref_mask = ((df_reference["driver_age"] >= low) & (df_reference["driver_age"] < high)).to_numpy()
    cur_mask = ((df_current["driver_age"] >= low) & (df_current["driver_age"] < high)).to_numpy()

    if actual_ref[ref_mask].sum() < 10 or actual_cur[cur_mask].sum() < 10:
        print(f"{label:<15} {'(insufficient claims)'}")
        continue

    g_ref_seg = gini_coefficient(actual_ref[ref_mask], pred_ref[ref_mask])
    g_cur_seg = gini_coefficient(actual_cur[cur_mask], pred_cur[cur_mask])

    seg_result = gini_drift_test(
        reference_gini=g_ref_seg,
        current_gini=g_cur_seg,
        reference_actual=actual_ref[ref_mask],
        reference_predicted=pred_ref[ref_mask],
        current_actual=actual_cur[cur_mask],
        current_predicted=pred_cur[cur_mask],
        n_bootstrap=200,
    )
    print(f"{label:<15} {seg_result.reference_gini:>10.4f}  {seg_result.current_gini:>10.4f}  "
          f"{seg_result.p_value:>10.4f}")
```

### Visualising Gini drift

Plotting the ROC curves for reference and current periods gives an immediate visual of whether and where discrimination has changed:

```python
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr_ref, tpr_ref, _ = roc_curve((actual_ref > 0).astype(int), pred_ref)
axes[0].plot(fpr_ref, tpr_ref, color="steelblue", linewidth=2,
             label=f"Reference (Gini={result.reference_gini:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC — Reference period")
axes[0].legend()

fpr_cur, tpr_cur, _ = roc_curve((actual_cur > 0).astype(int), pred_cur)
axes[1].plot(fpr_cur, tpr_cur, color="tomato", linewidth=2,
             label=f"Current (Gini={result.current_gini:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC — Current period")
axes[1].legend()

plt.suptitle(
    f"Gini drift: {result.reference_gini:.3f} → {result.current_gini:.3f}  "
    f"(p={result.p_value:.3f})",
    fontsize=13
)
plt.tight_layout()
plt.show()
```

Now we have all the individual metrics. Part 8 assembles them into a `MonitoringReport`.
