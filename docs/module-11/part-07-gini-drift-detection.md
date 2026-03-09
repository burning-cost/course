## Part 7: Gini drift detection

### What Gini drift measures

The Gini coefficient (also called the normalised Gini, or twice the AUC minus one) measures how well the model *ranks* risks. A Gini of 0 means the model has no discriminatory power - it ranks risks randomly. A Gini of 1 means the model ranks risks perfectly. In practice, a good UK motor frequency model achieves a Gini of around 0.35-0.50 on held-out data.

Gini drift is the question: has the Gini changed between the reference period and the current period? A falling Gini means the model is less able to distinguish high-risk from low-risk policies. This is a form of concept drift - the features that previously predicted claims well are no longer predicting them as well.

The `GiniDrift` test computes Ginis for both periods and runs a hypothesis test on the difference. The test statistic uses the DeLong variance estimator for the difference in AUCs, which properly accounts for the correlation structure when comparing two ROC curves.

### Why Gini drift is a different signal from A/E

The A/E ratio tests calibration: is the model's average prediction correct? Gini tests discrimination: does the model correctly rank risks? These can move independently.

A model can have a perfect A/E of 1.0 but falling Gini. This happens when the model's predictions are re-scaled correctly (so the average is right) but the ranking has weakened (so the ordering is less accurate). This is dangerous for pricing: the average premium may be correct, but the distribution of premiums across the portfolio may be wrong. High-risk policies may be under-priced; low-risk policies may be over-priced.

Conversely, a model can have a good Gini but a shifted A/E. The model still ranks risks correctly but at the wrong level. This is a calibration problem: apply a multiplicative adjustment to fix the average, and the discrimination is fine.

Understanding which type of drift you have determines the response:

- Calibration problem (A/E off, Gini OK): apply a recalibration factor
- Discrimination problem (Gini dropping, A/E may or may not be affected): retraining required

### Computing Gini drift

```python
from insurance_monitoring import GiniDrift

gini_calc = GiniDrift()

# We need binary claim outcomes (did a policy have a claim?)
# For multi-claim policies, treat any claim as 1
y_ref = (df_reference["claim_count"] > 0).to_numpy().astype(int)
y_cur = (df_current["claim_count"] > 0).to_numpy().astype(int)

gini_result = gini_calc.calculate(
    y_ref=y_ref,
    pred_ref=pred_ref,
    y_cur=y_cur,
    pred_cur=pred_cur,
)

print(f"Gini (reference): {gini_result.gini_ref:.4f}")
print(f"Gini (current):   {gini_result.gini_cur:.4f}")
print(f"Difference:       {gini_result.gini_cur - gini_result.gini_ref:+.4f}")
print(f"Z-statistic:      {gini_result.z_stat:.4f}")
print(f"P-value:          {gini_result.p_value:.4f}")
print(f"Traffic light:    {gini_result.traffic_light}")
```

`gini_ref` is the Gini computed on the reference period. `gini_cur` is the Gini computed on the current period. `z_stat` and `p_value` are from the DeLong test for equality of AUCs.

A p-value below 0.05 means the Gini difference is statistically significant - the model's discrimination has changed, and it is unlikely to be due to random variation.

### Interpreting the result

A statistically significant fall in Gini is serious. The intuition: if a model's Gini falls from 0.42 to 0.35, the model is less able to separate claimants from non-claimants. Risks that were in the top quintile of predicted frequency are now less likely to actually be high-frequency claims. The pricing order is becoming noisier.

Check whether the Gini fall is:

1. **Concentrated in a segment** - run the Gini calculation separately for different segments (age bands, regions) to identify where discrimination has fallen most
2. **Consistent across time** - if you have monthly data, compute Gini in rolling windows to see if it is trending down or was a one-off
3. **Correlated with a CSI-flagged feature** - if vehicle group has drifted significantly and Gini has fallen, a likely explanation is that vehicle group is a key discriminator and its relationship to risk has changed

```python
# Segment Gini analysis - run for age bands
print("\nGini by driver age band:")
print(f"{'Band':<15} {'Gini ref':>10}  {'Gini cur':>10}  {'p-value':>10}")
print("-" * 50)

for low, high, label in segments["driver_age_band"]:
    ref_mask = (df_reference["driver_age"] >= low) & (df_reference["driver_age"] < high)
    cur_mask = (df_current["driver_age"] >= low) & (df_current["driver_age"] < high)

    seg_y_ref = y_ref[ref_mask.to_numpy()]
    seg_pred_ref = pred_ref[ref_mask.to_numpy()]
    seg_y_cur = y_cur[cur_mask.to_numpy()]
    seg_pred_cur = pred_cur[cur_mask.to_numpy()]

    # Skip if not enough claims in either period
    if seg_y_ref.sum() < 10 or seg_y_cur.sum() < 10:
        print(f"{label:<15} {'(insufficient claims)'}")
        continue

    seg_result = gini_calc.calculate(
        y_ref=seg_y_ref,
        pred_ref=seg_pred_ref,
        y_cur=seg_y_cur,
        pred_cur=seg_pred_cur,
    )
    print(f"{label:<15} {seg_result.gini_ref:>10.4f}  {seg_result.gini_cur:>10.4f}  "
          f"{seg_result.p_value:>10.4f}")
```

### Visualising Gini drift

Plotting the ROC curves for reference and current periods side by side gives an immediate visual of whether and where discrimination has changed:

```python
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reference ROC
fpr_ref, tpr_ref, _ = roc_curve(y_ref, pred_ref)
axes[0].plot(fpr_ref, tpr_ref, color="steelblue", linewidth=2,
             label=f"Reference (Gini={gini_result.gini_ref:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC - Reference period")
axes[0].legend()

# Current ROC
fpr_cur, tpr_cur, _ = roc_curve(y_cur, pred_cur)
axes[1].plot(fpr_cur, tpr_cur, color="tomato", linewidth=2,
             label=f"Current (Gini={gini_result.gini_cur:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC - Current period")
axes[1].legend()

plt.suptitle(
    f"Gini drift: {gini_result.gini_ref:.3f} -> {gini_result.gini_cur:.3f}  "
    f"(p={gini_result.p_value:.3f})",
    fontsize=13
)
plt.tight_layout()
plt.savefig("/tmp/gini_drift.png", dpi=150, bbox_inches="tight")
plt.show()
```

Store the result:

```python
# Store for MonitoringReport
gini_drift = gini_result
```

Now we have all four metrics. Part 8 assembles them into a `MonitoringReport`.
