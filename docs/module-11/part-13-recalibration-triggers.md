## Part 13: Recalibration triggers

### The decision framework

When the monitoring report shows amber or red, you have three possible responses:

1. **Watch and wait** - the signal is not strong enough to act on yet
2. **Recalibrate** - adjust the model's output without retraining
3. **Retrain** - build a new model on updated data

These options are not equivalent in cost or risk. Watching and waiting costs nothing but risks sustained mis-pricing. Recalibration is fast (hours of analyst time) but only fixes the average, not the risk ordering. Retraining is thorough but takes weeks and requires a full validation and governance process.

The decision depends on which type of drift you have diagnosed (Part 9) and the magnitude of the signal.

### Trigger thresholds

These are the thresholds we recommend. They are not universally correct - your portfolio's claim volume, risk profile, and governance standards all affect where the thresholds should sit. Treat these as defaults that your pricing governance committee should sign off.

**Recalibration trigger:**

| Condition | Action |
|-----------|--------|
| A/E ratio 95% CI excludes 1.0 for two consecutive months | Recalibrate |
| A/E point estimate outside [0.90, 1.10] in any single month | Recalibrate |
| Score PSI > 0.20 and A/E CI excludes 1.0 | Recalibrate |

**Retraining trigger:**

| Condition | Action |
|-----------|--------|
| Statistically significant Gini drop (p < 0.05, drop > 0.03) for two consecutive months | Retrain |
| A/E outside [0.85, 1.15] for any single month | Retrain |
| Recalibration factor has been applied twice without resolving the A/E signal | Retrain |

**Escalation trigger (head of pricing + actuarial function):**

| Condition | Action |
|-----------|--------|
| Overall RED for any single month | Immediate escalation |
| Three consecutive AMBER months | Scheduled review, retrain timeline agreed |

### Implementing recalibration

Recalibration applies a multiplicative factor to the model's frequency predictions to restore the A/E to 1.0. It is the simplest possible correction:

```python
def compute_recalibration_factor(
    actual: np.ndarray,
    expected: np.ndarray,
    exposure: np.ndarray,
) -> float:
    """
    Compute the recalibration factor as the inverse of the A/E ratio.

    Multiply model predictions by this factor to restore portfolio-level calibration.
    """
    ae_result = ae_calc.calculate(actual=actual, expected=expected, exposure=exposure)
    factor = 1.0 / ae_result.ratio
    print(f"A/E ratio:           {ae_result.ratio:.4f}")
    print(f"Recalibration factor: {factor:.4f}")
    print(f"Interpretation: multiply all predictions by {factor:.4f}")
    return factor


recal_factor = compute_recalibration_factor(
    actual=actual_cur,
    expected=expected_cur,
    exposure=exposure_cur,
)
```

Applying the factor in the pricing pipeline:

```python
# In the pricing pipeline (Module 8), after model prediction:
# predicted_frequency = model.predict(X) * recal_factor
#
# The recal_factor is stored in a configuration table:

recal_record = {
    "model_name":          MODEL_NAME,
    "model_version":       MODEL_VERSION,
    "effective_from":      CURRENT_DATE,
    "recalibration_factor": float(recal_factor),
    "ae_ratio_at_trigger": float(ae_portfolio.ratio),
    "applied_by":          "automated_monitoring",
    "monitoring_run_date": RUN_DATE,
    "reason":              "A/E ratio outside [0.95, 1.05] for two consecutive months",
}

recal_df = spark.createDataFrame([Row(**recal_record)])

recal_table = f"{CATALOG}.{SCHEMA}.recalibration_history"

(
    recal_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(recal_table)
)

print(f"Recalibration factor {recal_factor:.4f} logged to {recal_table}")
```

### Why the recalibration factor is not enough

A recalibration factor adjusts the overall level of predictions but changes nothing about their relative ordering. If the model is under-predicting young drivers by 30% and over-predicting older drivers by 20%, applying a single multiplicative factor of 1.08 will make the portfolio average look right while still mispredicting both segments.

This is why recalibration is a holding measure, not a solution. It buys time while you assess whether full retraining is necessary. It should always be accompanied by a documented timeline for the retraining review.

### Automated recalibration decision in the monitoring notebook

Add this block at the end of the monitoring notebook to compute and log the recalibration recommendation automatically:

```python
def recalibration_recommendation(
    ae_result,
    gini_result,
    monitoring_log_df,
) -> dict:
    """
    Determine the monitoring recommendation based on current metrics
    and recent history.
    """
    ae_ci_excludes_1 = (ae_result.ci_lower > 1.0) or (ae_result.ci_upper < 1.0)
    ae_outside_10pct = (ae_result.ratio < 0.90) or (ae_result.ratio > 1.10)
    ae_outside_15pct = (ae_result.ratio < 0.85) or (ae_result.ratio > 1.15)
    gini_significant = (gini_result.p_value < 0.05) and (
        abs(gini_result.gini_cur - gini_result.gini_ref) >= 0.03
    )

    # Check how many consecutive months the CI has excluded 1.0
    # consecutive_amber counts prior history rows (not the current month).
    # "Two consecutive months" = current month + one prior month = consecutive_amber >= 1.
    consecutive_amber = 0
    if monitoring_log_df is not None:
        recent = (
            monitoring_log_df
            .sort("current_date", descending=True)
            .head(3)
        )
        for row in recent.iter_rows(named=True):
            if row["ae_ci_lower"] > 1.0 or row["ae_ci_upper"] < 1.0:
                consecutive_amber += 1
            else:
                break

    if ae_outside_15pct:
        return {"recommendation": "RETRAIN", "reason": "A/E outside [0.85, 1.15]"}
    elif gini_significant:
        return {"recommendation": "RETRAIN", "reason": "Statistically significant Gini drop"}
    elif ae_outside_10pct:
        return {"recommendation": "RECALIBRATE", "reason": "A/E outside [0.90, 1.10]"}
    elif ae_ci_excludes_1 and consecutive_amber >= 1:
        return {"recommendation": "RECALIBRATE", "reason": "A/E CI excludes 1.0 for 2+ months"}
    elif ae_ci_excludes_1:
        return {"recommendation": "WATCH", "reason": "A/E CI excludes 1.0 - monitor next month"}
    else:
        return {"recommendation": "NO_ACTION", "reason": "All signals within tolerance"}


# Load recent monitoring history
try:
    recent_history = pl.from_pandas(
        spark.sql(f"""
            SELECT current_date, ae_ratio, ae_ci_lower, ae_ci_upper
            FROM {TABLES['monitoring_log']}
            WHERE model_name = '{MODEL_NAME}'
            ORDER BY current_date DESC
            LIMIT 3
        """).toPandas()
    )
except Exception:
    recent_history = None  # First run, no history yet

recommendation = recalibration_recommendation(
    ae_portfolio,
    gini_drift,
    recent_history,
)

print(f"\nRECOMMENDATION: {recommendation['recommendation']}")
print(f"Reason: {recommendation['reason']}")
```

This gives you a machine-readable recommendation that can be logged to the monitoring table and included in the automated alert.
