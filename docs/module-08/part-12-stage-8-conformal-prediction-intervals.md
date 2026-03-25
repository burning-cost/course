## Part 12: Stage 8 — Conformal prediction intervals

The conformal predictor quantifies uncertainty in the severity model's predictions. We covered the theory in Module 5. The critical requirement here is that calibration and test data are exchangeable — which in insurance means the split must be temporal, not random. We use accident year 2024 as the calibration set and accident year 2025 as the test set.

Add a markdown cell:

```python
%md
## Stage 8: Conformal prediction intervals
```

### Setting up the calibration split

```python
from insurance_conformal import InsuranceConformalPredictor

# Calibration set: accident year 2024 (most recent training year, claims-only)
# Test set: accident year 2025 (held-out, claims-only)
#
# The severity model was trained on 2022-2023 (Stage 6), so the 2024 calibration
# set is genuinely out-of-sample for the severity model. This is required for the
# conformal coverage guarantee to hold.

df_cal_sev = features_pd[
    (features_pd["accident_year"] == cal_year) &
    (features_pd["claim_count"] > 0)
].copy()
df_te_sev = df_test[df_test["claim_count"] > 0].copy()

df_cal_sev["mean_sev"] = df_cal_sev["incurred_loss"] / df_cal_sev["claim_count"]
df_te_sev["mean_sev"]  = df_te_sev["incurred_loss"]  / df_te_sev["claim_count"]

X_cal = df_cal_sev[FEATURE_COLS]
y_cal = df_cal_sev["mean_sev"].values

X_te_conf = df_te_sev[FEATURE_COLS]
y_te_conf = df_te_sev["mean_sev"].values

print(f"Calibration year:  {cal_year}")
print(f"Calibration claims:{len(df_cal_sev):,}")
print(f"Test claims:       {len(df_te_sev):,}")

# The finite-sample correction to conformal coverage is 1/(n_cal + 1).
# For n_cal=1000, the correction is 0.001. For n_cal=100, it is 0.01.
# Below 50 claims, intervals are unreliable — the correction becomes material.
if len(df_cal_sev) < 100:
    print(f"WARNING: {len(df_cal_sev)} calibration claims is borderline. "
          f"Consider a wider calibration window.")
else:
    print(f"Calibration set size: adequate.")
```

### Calibrating the conformal predictor

```python
cp = InsuranceConformalPredictor(
    model=sev_model,
    nonconformity="pearson_weighted",   # (y - yhat) / sqrt(yhat^p): Module 5 explained why
    distribution="tweedie",
    tweedie_power=2.0,                  # Gamma (variance power = 2)
)
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals (alpha=0.10) for the test set
intervals = cp.predict_interval(X_te_conf, alpha=CONFORMAL_ALPHA)

# intervals: DataFrame with columns lower, point, upper
print(f"\nSeverity prediction intervals (1-alpha = {1-CONFORMAL_ALPHA:.0%}):")
print(f"  Mean lower: £{intervals['lower'].mean():,.0f}")
print(f"  Mean point: £{intervals['point'].mean():,.0f}")
print(f"  Mean upper: £{intervals['upper'].mean():,.0f}")
print(f"  Mean width: £{(intervals['upper'] - intervals['lower']).mean():,.0f}")
print(f"  Width / point: {((intervals['upper'] - intervals['lower']) / intervals['point']).mean():.2f}")
```

### Coverage validation

Coverage must be validated by decile, not just overall. A conformal predictor that achieves 90% overall coverage but 72% coverage in the top decile of predicted severity is unsafe for underwriting referral — the policies where uncertainty matters most are the ones where the guarantee fails.

```python
diag = cp.coverage_by_decile(X_te_conf, y_te_conf, alpha=CONFORMAL_ALPHA)
min_cov = float(diag["coverage"].min())

print(f"\nCoverage by predicted severity decile (target {1-CONFORMAL_ALPHA:.0%}):")
print(f"{'Decile':<8} {'N claims':>10} {'Coverage':>10} {'Status':>14}")
for row in diag.iter_rows(named=True):
    status = "OK" if row["coverage"] >= 1 - CONFORMAL_ALPHA - 0.05 else "WARN"
    print(f"  {row['decile']:<6} {row.get('n', '?'):>10} "
          f"{row['coverage']:>10.3f} {status:>14}")

print(f"\nMin decile coverage: {min_cov:.3f}")
if min_cov < 0.85:
    print("WARNING: Coverage below 85% in at least one decile.")
    print("Possible causes:")
    print("  1. Calibration set too small (< 500 claims)")
    print("  2. Distribution shift between calibration and test year")
    print("     (7% inflation per year is large — check severity trends)")
    print("  3. Nonconformity score mismatch with severity distribution tail")
    print("Consider: wider calibration window (use 2023-2024 combined).")
else:
    print("Coverage acceptable. Intervals may be used for underwriting referral.")
```

### Flagging high-uncertainty risks

```python
# Relative interval width = (upper - lower) / point
# High relative width = high model uncertainty for this individual risk
rel_width = (intervals["upper"] - intervals["lower"]) / (intervals["point"] + 1e-6)

# Top 10% of relative width: flag for underwriting referral
referral_threshold = float(np.percentile(rel_width, 90))
referral_flag      = rel_width >= referral_threshold

print(f"\nUnderwriting referral:")
print(f"  Relative width threshold (P90): {referral_threshold:.2f}")
print(f"  Flagged for referral:           {int(referral_flag.sum()):,} "
      f"({referral_flag.mean():.1%} of claims set)")
```

### Writing conformal intervals to Delta

```python
conf_df = pl.DataFrame({
    "policy_id":        df_te_sev["policy_id"].tolist(),
    "accident_year":    df_te_sev["accident_year"].tolist(),
    "mean_sev_actual":  y_te_conf.tolist(),
    "sev_lower":        intervals["lower"].to_list(),
    "sev_point":        intervals["point"].to_list(),
    "sev_upper":        intervals["upper"].to_list(),
    "rel_width":        rel_width.to_list(),
    "referral_flag":    referral_flag.to_list(),
    "conformal_alpha":  [CONFORMAL_ALPHA] * len(df_te_sev),
    "cal_year":         [cal_year]        * len(df_te_sev),
    "sev_run_id":       [sev_run_id]      * len(df_te_sev),
    "run_date":         [RUN_DATE]        * len(df_te_sev),
})

(
    spark.createDataFrame(conf_df.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["conformal_intervals"])
)

spark.sql(f"""
    ALTER TABLE {TABLES['conformal_intervals']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

print(f"Conformal intervals written: {len(conf_df):,} rows → {TABLES['conformal_intervals']}")
```

**Why Pearson-weighted nonconformity?** The Pearson residual is `(y - yhat) / sqrt(yhat^p)` where `p=2` for the Gamma distribution. Raw residuals `(y - yhat)` are heteroscedastic for severity data — large claims have proportionally larger residuals than small claims. The Pearson residual normalises for this, producing a score that is closer to homoscedastic across the severity range. Module 5 showed that this reduces interval width by roughly 25-30% relative to raw residuals, with identical coverage guarantees.
