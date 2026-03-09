## Part 12: Stage 8 -- Conformal prediction intervals

The conformal predictor quantifies uncertainty in the severity predictions. We covered the theory in Module 5. The critical requirement for a valid conformal interval is that calibration and test data are exchangeable -- which means the split must be temporal, not random.

Add a markdown cell:

```python
%md
## Stage 8: Conformal prediction intervals -- temporal calibration split
```

```python
from insurance_conformal import InsuranceConformalPredictor

# -----------------------------------------------------------------------
# Temporal calibration split.
#
# Module 5 established that calibration and test data must be exchangeable.
# Exchangeability requires that they come from the same distribution.
# A random split violates this for insurance data because time trends
# (inflation, mix changes, legislative effects) mean 2022 observations
# and 2023 observations are NOT exchangeable.
#
# Correct approach: use the penultimate accident year as calibration
# and the final accident year as test. Both are out-of-time relative
# to the training set.
# -----------------------------------------------------------------------

# Identify the calibration year: second most recent accident year
all_years   = sorted(df_train["accident_year"].unique())
cal_year    = all_years[-1]   # most recent training year = calibration set
print(f"Calibration year: {cal_year}")
print(f"Test year:        {max_year}")

# Claims-only sets for conformal calibration
# Conformal intervals are calibrated on severity predictions
df_cal_sev = features_pd[
    (features_pd["accident_year"] == cal_year) &
    (features_pd["claim_count"] > 0)
].copy()
df_te_sev  = df_test[df_test["claim_count"] > 0].copy()

df_cal_sev["mean_sev"] = df_cal_sev["claim_amount"] / df_cal_sev["claim_count"]
df_te_sev["mean_sev"]  = df_te_sev["claim_amount"]  / df_te_sev["claim_count"]

X_cal = df_cal_sev[FEATURE_COLS]
y_cal = df_cal_sev["mean_sev"].values

X_te_conf = df_te_sev[FEATURE_COLS]
y_te_conf = df_te_sev["mean_sev"].values

print(f"Calibration claims: {len(df_cal_sev):,}")
print(f"Test claims:        {len(df_te_sev):,}")

# -----------------------------------------------------------------------
# A note on calibration set size.
# The conformal coverage guarantee has a finite-sample correction:
# P(y_new inside interval) >= (1 - alpha) - 1/(n_cal + 1)
# For n_cal=1000, the correction is 0.001 -- negligible.
# For n_cal=100, it is 0.01 -- still acceptable.
# For n_cal<50, the intervals are unreliable. If your calibration set
# has fewer than 50 claims, reconsider the temporal split.
# -----------------------------------------------------------------------
if len(df_cal_sev) < 100:
    print(f"WARNING: Calibration set has only {len(df_cal_sev)} claims. "
          f"Conformal intervals may be unreliable. Consider a wider calibration window.")
```

### Calibrating the conformal predictor

```python
cp = InsuranceConformalPredictor(
    model=sev_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=2.0,
)
cp.calibrate(X_cal, y_cal)

# Generate 90% prediction intervals for the test set
intervals = cp.predict_interval(X_te_conf, alpha=CONFORMAL_ALPHA)

# intervals is a DataFrame with columns: lower, point, upper
# lower: lower bound of the 90% interval
# point: point prediction (same as sev_model.predict())
# upper: upper bound of the 90% interval

print(f"\nSeverity prediction intervals (90%, alpha={CONFORMAL_ALPHA}):")
print(f"  Mean lower bound: £{intervals['lower'].mean():,.0f}")
print(f"  Mean point pred:  £{intervals['point'].mean():,.0f}")
print(f"  Mean upper bound: £{intervals['upper'].mean():,.0f}")
print(f"  Mean interval width: £{(intervals['upper'] - intervals['lower']).mean():,.0f}")
```

### Coverage validation

The coverage check is mandatory. A conformal predictor that achieves 90% overall coverage but only 72% coverage in the top decile of risks is not usable for reserving or underwriting referral.

```python
diag = cp.coverage_by_decile(X_te_conf, y_te_conf, alpha=CONFORMAL_ALPHA)
min_cov = float(diag["coverage"].min())

print("\nCoverage by predicted severity decile:")
print(f"{'Decile':<8} {'Coverage':>10} {'Status':>12}")
for row in diag.iter_rows(named=True):
    status = "OK" if row["coverage"] >= 0.85 else "WARN: below 85%"
    print(f"  {row['decile']:<6} {row['coverage']:>10.3f} {status:>12}")

print(f"\nOverall min decile coverage: {min_cov:.3f}")
if min_cov < 0.85:
    print("WARNING: Coverage below 85% in at least one decile.")
    print("Possible causes:")
    print("  1. Calibration set is too small (fewer than ~500 claims)")
    print("  2. Claims inflation or mix change between calibration and test years")
    print("  3. Nonconformity score specification does not match the severity distribution")
    print("  4. The calibration and test distributions are genuinely different")
    print("Consider: wider calibration window, or recalibrate on more recent data.")
else:
    print("Coverage is acceptable in all deciles. Intervals may be used for "
          "underwriting referral and minimum premium floor setting.")
```

### Flagging uncertain risks for underwriting referral

```python
# Relative interval width = (upper - lower) / point
# High relative width = high model uncertainty for this risk
rel_width = (intervals["upper"] - intervals["lower"]) / (intervals["point"] + 1e-6)

# Flag risks in the top 10% of relative width for underwriting referral
referral_threshold = np.percentile(rel_width, 90)
referral_flag      = rel_width >= referral_threshold

print(f"\nUnderwriting referral flag:")
print(f"  Relative width threshold (P90): {referral_threshold:.2f}")
print(f"  Policies flagged for referral:  {referral_flag.sum():,} "
      f"({referral_flag.mean():.1%} of claims set)")
```

### Writing conformal intervals to Delta

```python
conf_df = pl.DataFrame({
    "policy_id":    df_te_sev["policy_id"].tolist(),
    "accident_year": df_te_sev["accident_year"].tolist(),
    "mean_sev_actual": y_te_conf.tolist(),
    "sev_lower":    intervals["lower"].to_list(),
    "sev_point":      intervals["point"].to_list(),
    "sev_upper":    intervals["upper"].to_list(),
    "rel_width":    rel_width.to_list(),
    "referral_flag": referral_flag.to_list(),
    "freq_run_id":  [freq_run_id] * len(df_te_sev),
    "sev_run_id":   [sev_run_id]  * len(df_te_sev),
    "run_date":     [RUN_DATE]    * len(df_te_sev),
})

spark.createDataFrame(conf_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["conformal_intervals"])

print(f"Conformal intervals written: {len(conf_df):,} rows")
print(f"Table: {TABLES['conformal_intervals']}")
```