# Module 5 Exercises: Conformal Prediction Intervals

Five exercises. Work through them in order - each builds on the previous. If you get stuck, read the hint before looking at the solution. The goal is to understand what the code is doing, not just to make it run.

---

## Before you start

These exercises use the model and data you built in the tutorial. If you closed your notebook, you will need to regenerate the data and refit the model. The simplest approach: add a new cell at the top of your tutorial notebook and run the exercises there. The model, conformal predictor, and test set are already in memory.

If you are starting fresh in a new notebook, run the setup cell below first.

**Setup for a fresh notebook (skip this if your tutorial notebook is still running):**

```python
%pip install "insurance-conformal[catboost]" catboost polars mlflow --quiet
```

```python
dbutils.library.restartPython()
```

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import date
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor
from insurance_datasets import load_motor

# Load the same dataset as the tutorial
df = pl.from_pandas(load_motor(n_policies=100_000, seed=42))

# Add synthetic accident years for the temporal split (same approach as tutorial Part 4)
rng_year = np.random.default_rng(seed=42)
accident_year = rng_year.choice([2019, 2020, 2021, 2022, 2023], size=len(df),
                                 p=[0.15, 0.17, 0.20, 0.23, 0.25])
df = df.with_columns(
    pl.Series("accident_year", accident_year.astype(np.int32))
).sort("accident_year").with_columns(
    (pl.col("claim_amount") / pl.col("exposure")).alias("pure_premium")
)

X_COLS       = ["age", "vehicle_age", "vehicle_group", "region", "credit_score"]
CAT_FEATURES = ["region"]
n_df         = len(df)
train_end    = int(0.60 * n_df)
cal_end      = int(0.80 * n_df)

X_train = df[:train_end][X_COLS].to_pandas()
y_train = df[:train_end]["pure_premium"].to_pandas()
X_cal   = df[train_end:cal_end][X_COLS].to_pandas()
y_cal   = df[train_end:cal_end]["pure_premium"].to_pandas()
X_test  = df[cal_end:][X_COLS].to_pandas()
y_test  = df[cal_end:]["pure_premium"].to_pandas()

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
cal_pool   = Pool(X_cal, y_cal, cat_features=CAT_FEATURES)
test_pool  = Pool(X_test, y_test, cat_features=CAT_FEATURES)

model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    eval_metric="Tweedie:variance_power=1.5",
    learning_rate=0.05,
    depth=5,
    min_data_in_leaf=50,
    iterations=500,
    random_seed=42,
    verbose=0,
)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

print("Setup complete. Model fitted and conformal predictor calibrated.")
print(f"Training rows: {len(X_train):,}, Calibration rows: {len(X_cal):,}, Test rows: {len(X_test):,}")
```

---

## Exercise 1: Coverage-by-decile - Pearson weighted vs raw residuals

**Objective.** Run the coverage-by-decile diagnostic on both `pearson_weighted` and raw residual intervals. See directly why the raw score fails for insurance data and what the Pearson score fixes.

**Business context.** Your head of reserving wants to use the upper bounds of prediction intervals as range inputs for the next reserve cycle. Before you hand anything over, you need to demonstrate that the intervals achieve stated coverage across the full risk spectrum - not just on average. The finance director has seen a competitor firm's reserving team produce reserve ranges that were too narrow for the top risk decile, resulting in a reserve strengthening announcement mid-year. You need to show this will not happen here.

**What to do:**

1. Create two conformal predictors: one with `nonconformity="pearson_weighted"` and one with `nonconformity="raw"`. Calibrate both on `X_cal, y_cal`. Generate 90% prediction intervals on the test set from each.

2. Run `coverage_by_decile(X_test, y_test, alpha=0.10)` on both predictors. Print the results in a side-by-side table. What is the coverage in the top risk decile for each? What is it in the bottom decile?

3. Compute the mean and median interval width for each approach. Does the raw score produce narrower or wider intervals? Is the width difference larger in the low-risk or high-risk deciles?

4. In one sentence, state what would happen if your reserving team used raw residual intervals as the 90% upper bound for the top decile of risks.

5. **Extension (optional):** Try `nonconformity="deviance"` as a third option. How does it compare to Pearson-weighted on coverage-by-decile?

**Start here - set up both predictors:**

```python
# Exercise 1: Create predictors with different non-conformity scores
scores_to_test = [
    ("pearson_weighted", "Pearson weighted"),
    ("raw",              "Raw residual"),
]

results = {}

for score_name, label in scores_to_test:
    cp_ex1 = InsuranceConformalPredictor(
        model=model,
        nonconformity=score_name,
        distribution="tweedie",
        tweedie_power=1.5,
    )
    cp_ex1.calibrate(X_cal, y_cal)
    intervals = cp_ex1.predict_interval(X_test, alpha=0.10)
    diag      = cp_ex1.coverage_by_decile(X_test, y_test, alpha=0.10)

    results[label] = {
        "predictor": cp_ex1,
        "intervals": intervals,
        "diag":      diag,
    }
    print(f"Calibrated and generated intervals for: {label}")

print("\nBoth predictors ready. Now complete Tasks 2-4.")
```

<details>
<summary>Hint for Task 2: why does raw fail?</summary>

The raw residual score normalises nothing. A miss of £200 on a £200-risk and a miss of £200 on a £20,000-risk produce identical scores. The calibration quantile is set on all calibration observations. The majority of calibration observations are low-risk policies (they numerically dominate any insurance portfolio), so the 90th percentile absolute residual reflects the scale of low-risk residuals.

When that fixed-width threshold is applied to high-risk policies - whose absolute residuals are genuinely much larger because the losses themselves are larger - the interval is too narrow. The top decile's actual outcomes exceed the interval far more than 10% of the time.

</details>

<details>
<summary>Solution - Exercise 1</summary>

```python
import numpy as np

# Task 2: Side-by-side coverage table
labels = list(results.keys())
diags  = [results[l]["diag"] for l in labels]

print(f"{'Decile':<8}", end="")
for label in labels:
    print(f"{label:>22}", end="")
print()
print("-" * (8 + 22 * len(labels)))

for decile_idx in range(10):
    print(f"{decile_idx + 1:<8}", end="")
    for label in labels:
        cov = results[label]["diag"]["coverage"].to_list()[decile_idx]
        print(f"{cov:>22.3f}", end="")
    print()

print()
for label in labels:
    d         = results[label]["diag"]["coverage"].to_list()
    print(f"{label}:")
    print(f"  Bottom decile (decile 1): {d[0]:.3f}")
    print(f"  Top decile (decile 10):   {d[-1]:.3f}")
    print(f"  Spread (max - min):       {max(d) - min(d):.3f}")

# Task 3: Interval widths
print()
print(f"{'Approach':<25} {'Mean width':>12} {'Median width':>14}")
print("-" * 54)
for label in labels:
    ivs    = results[label]["intervals"]
    widths = (ivs["upper"] - ivs["lower"]).to_numpy()
    print(f"{label:<25} {widths.mean():>12.2f} {np.median(widths):>14.2f}")

# Task 3 continued: width by decile
# Bin the test set by predicted pure premium
preds_test = model.predict(test_pool)
decile_bin = pl.Series(
    np.digitize(preds_test, np.quantile(preds_test, np.linspace(0, 1, 11)), right=False)
    .clip(1, 10)
)

print("\nMean interval width by decile:")
print(f"{'Decile':<8}", end="")
for label in labels:
    print(f"{label:>22}", end="")
print()

for d in range(1, 11):
    mask = decile_bin.to_numpy() == d
    print(f"{d:<8}", end="")
    for label in labels:
        ivs   = results[label]["intervals"]
        w     = (ivs["upper"] - ivs["lower"]).to_numpy()
        mean_w = w[mask].mean()
        print(f"{mean_w:>22.2f}", end="")
    print()

# Task 4: Reserve danger
print("""
Task 4 answer:
If raw residual intervals are used as reserve range inputs for the top risk decile,
the reserving team would face adverse outcomes roughly one time in four (coverage ~75%)
rather than the stated one time in ten (coverage 90%), systematically understating
reserve requirements for the book's largest loss contributors.
""")

# Extension: deviance score
cp_dev = InsuranceConformalPredictor(
    model=model,
    nonconformity="deviance",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp_dev.calibrate(X_cal, y_cal)
diag_dev = cp_dev.coverage_by_decile(X_test, y_test, alpha=0.10)
cov_dev  = diag_dev["coverage"].to_list()
print(f"Deviance score: bottom decile {cov_dev[0]:.3f}, top decile {cov_dev[-1]:.3f}, spread {max(cov_dev)-min(cov_dev):.3f}")
```

**What you should see:** the raw residual approach has coverage below 80% in the top decile (often 72-78%) while marginal coverage is near 90%. Pearson weighted achieves much flatter coverage across deciles. The raw approach produces similar mean interval widths overall, but the widths are too uniform - they do not scale with risk level the way the Pearson approach does.

</details>

---

## Exercise 2: Calibration set size and the coverage confidence interval

**Objective.** Verify empirically that coverage precision improves with calibration set size, and determine the minimum calibration set your team should insist on.

**Business context.** Your book has 8,000 policies in the most recent accident year - the candidate calibration set. Your chief actuary wants to keep 6,000 for additional training data and use only 2,000 for calibration. Your statistician says the full 8,000 should be used. You need a defensible recommendation backed by data.

**Background on what you are measuring:**

There are two distinct notions of precision here, and you should not confuse them:

- **The formal conformal guarantee**: for a calibration set of size `n`, coverage is at least `1 - alpha - 1/(n+1)`. For n=2,000, the correction is 0.0005 - negligible. The formal guarantee is not the binding constraint.

- **The precision of your coverage estimate**: even if the true coverage is exactly 90%, your empirical test set coverage will vary due to sampling noise. The standard deviation of an empirical coverage estimate over a test set of size `m` is approximately `sqrt(alpha*(1-alpha)/m)`. For m=20,000 and alpha=0.10, this is about 0.2pp. For m=2,000, it is about 0.7pp. This is testing precision, not calibration precision.

- **The effect of calibration size on actual coverage**: more calibration observations give a more precise estimate of the `1-alpha` quantile of the non-conformity score. A less precise quantile estimate means actual coverage will vary more from run to run. The standard deviation of the quantile estimate scales as approximately `1/sqrt(n_cal)`.

Exercise 2 measures the third quantity: how much does coverage vary across bootstrap samples of the calibration set?

**What to do:**

1. For calibration set sizes `n_cal` = [200, 500, 1000, 2000, 5000], run 20 bootstrap repetitions: draw `n_cal` observations (with replacement) from `X_cal, y_cal`, calibrate a `pearson_weighted` predictor, generate 90% intervals on the full test set, compute empirical coverage. Report the mean and standard deviation of coverage for each `n_cal`.

2. Does the standard deviation of coverage decrease roughly as `1/sqrt(n_cal)`? Verify this by computing the ratio `SD * sqrt(n_cal)` - if the relationship holds, this ratio should be approximately constant.

3. At what `n_cal` does the 95% confidence interval for coverage (mean ± 1.96 × SD) first stay entirely within [87%, 93%]? This is your minimum calibration set recommendation.

4. Your book has 50,000 total policies. You must hold out 20% (10,000) as a genuine test set. Of the remaining 40,000, how should you split between training and calibration to maximise out-of-sample prediction performance while meeting your minimum calibration requirement? Give a specific recommendation and justify it.

**Start here:**

```python
# Exercise 2: Bootstrap across calibration set sizes
import time

cal_sizes = [200, 500, 1_000, 2_000, 5_000]
n_boot    = 20
rng_boot  = np.random.default_rng(seed=99)

X_cal_arr = X_cal.reset_index(drop=True)
y_cal_arr = y_cal.to_numpy()
y_test_arr = y_test.to_numpy()

print(f"Running {n_boot} bootstrap replicates for each of {len(cal_sizes)} calibration sizes...")
print("This takes 3-5 minutes. Read the hints while you wait.\n")
```

<details>
<summary>Hint for Task 1: how to bootstrap calibration</summary>

For each bootstrap replicate:
1. Draw a random index array of size `n_cal` from `range(len(X_cal))`, with replacement
2. Subset `X_cal` and `y_cal` using those indices: `X_sub = X_cal_arr.iloc[idx]`
3. Create a fresh `InsuranceConformalPredictor`, calibrate it on `X_sub, y_sub`
4. Generate intervals on the full `X_test`
5. Compute coverage: `mean((y_test >= lower) & (y_test <= upper))`

The `Pool` for calibration should use `cat_features=CAT_FEATURES`. The `X_sub` is a pandas DataFrame, which the predictor accepts directly.

</details>

<details>
<summary>Hint for Task 3: finding the minimum n_cal</summary>

For each `n_cal`, compute:
- `mean_coverage` = mean of the 20 coverage values
- `sd_coverage`   = standard deviation of the 20 coverage values
- `ci_lo = mean_coverage - 1.96 * sd_coverage`
- `ci_hi = mean_coverage + 1.96 * sd_coverage`

The condition is: `ci_lo >= 0.87 AND ci_hi <= 0.93`. The first `n_cal` where both conditions hold simultaneously is your recommendation.

You can also derive it theoretically: SD ≈ `c / sqrt(n_cal)` where `c` is approximately constant. If you fit `c` from the bootstrap results, you can predict the n_cal needed for any target precision.

</details>

<details>
<summary>Solution - Exercise 2</summary>

```python
import numpy as np
import time

X_cal_arr  = X_cal.reset_index(drop=True)
y_cal_arr  = y_cal.to_numpy()
y_test_arr = y_test.to_numpy()

cal_sizes = [200, 500, 1_000, 2_000, 5_000]
n_boot    = 20
rng_boot  = np.random.default_rng(seed=99)

summary = []
for n_cal in cal_sizes:
    coverages = []
    t0 = time.time()
    for _ in range(n_boot):
        idx   = rng_boot.choice(len(X_cal_arr), size=n_cal, replace=True)
        X_sub = X_cal_arr.iloc[idx]
        y_sub = y_cal_arr[idx]

        cp_boot = InsuranceConformalPredictor(
            model=model,
            nonconformity="pearson_weighted",
            distribution="tweedie",
            tweedie_power=1.5,
        )
        cp_boot.calibrate(X_sub, y_sub)
        ivs     = cp_boot.predict_interval(X_test, alpha=0.10)
        covered = (
            (y_test_arr >= ivs["lower"].to_numpy()) &
            (y_test_arr <= ivs["upper"].to_numpy())
        )
        coverages.append(covered.mean())

    elapsed      = time.time() - t0
    mean_cov     = np.mean(coverages)
    sd_cov       = np.std(coverages)
    ci_lo        = mean_cov - 1.96 * sd_cov
    ci_hi        = mean_cov + 1.96 * sd_cov
    within_3pp   = (ci_lo >= 0.87) and (ci_hi <= 0.93)
    sd_x_sqrtn   = sd_cov * np.sqrt(n_cal)   # should be approximately constant

    summary.append({
        "n_cal":      n_cal,
        "mean_cov":   round(mean_cov, 4),
        "sd_cov":     round(sd_cov, 5),
        "ci_lo":      round(ci_lo, 4),
        "ci_hi":      round(ci_hi, 4),
        "within_3pp": within_3pp,
        "sd_x_sqrtn": round(sd_x_sqrtn, 4),
    })
    print(f"n_cal={n_cal:5d}: mean={mean_cov:.3f}  SD={sd_cov:.5f}  "
          f"95%CI=[{ci_lo:.3f},{ci_hi:.3f}]  within_3pp={within_3pp}  "
          f"SD*sqrt(n)={sd_x_sqrtn:.4f}  ({elapsed:.0f}s)")

# Task 2: Is SD * sqrt(n) approximately constant?
print("\nTask 2: SD × sqrt(n_cal) (should be approximately constant if 1/sqrt(n) relationship holds)")
for row in summary:
    print(f"  n_cal={row['n_cal']:5d}: SD × sqrt(n) = {row['sd_x_sqrtn']:.4f}")

# Task 3: Minimum n_cal
min_n_cal = None
for row in summary:
    if row["within_3pp"]:
        min_n_cal = row["n_cal"]
        print(f"\nTask 3: Minimum n_cal where 95% CI stays within [87%, 93%]: {min_n_cal}")
        break
if min_n_cal is None:
    print("\nTask 3: None of the tested sizes achieve the [87%, 93%] criterion.")
    print("Fit a theoretical curve: SD ≈ c/sqrt(n), solve for n when 1.96*SD = 0.03.")

# Task 4: Split recommendation
# 50,000 total policies. 10,000 held for test (20%).
# Remaining 40,000 to split between training and calibration.
# Minimum calibration: ~2,000 from Task 3.
# GBM sample efficiency: training benefit is greatest going from 10,000 to 30,000 rows;
# returns diminish beyond 35,000-40,000. Adding 3,000 calibration rows (5,000 total)
# adds negligible model improvement vs the precision gain in coverage estimation.
print("""
Task 4 recommendation: 35,000 training / 5,000 calibration / 10,000 test.

Rationale:
- 5,000 calibration observations gives SD ~ 0.013, so 95% CI is [0.87-0.90, 0.90-0.93]:
  comfortably within the 3pp tolerance.
- The marginal value of 3,000 additional calibration observations (going from 5,000 to 8,000)
  reduces coverage SD from ~0.013 to ~0.011 - a modest improvement.
- The marginal value of 3,000 additional training observations (going from 35,000 to 38,000)
  improves the base model's generalisation, particularly in thin cells where intervals
  are widest. Improving the base model is more valuable than marginal coverage precision.
- Bottom line: 35k/5k is the right split. Only use more calibration if your test coverage
  estimates show substantial run-to-run variation in production.
""")
```

</details>

---

## Exercise 3: Uncertain risk flagging for underwriting referral

**Objective.** Build the underwriting referral flag using relative interval width. Characterise what gets flagged, verify the flag is stable across portfolio segments, and prepare the explanation for your underwriting director.

**Business context.** Your underwriting director has agreed to a 10% referral rate: one in ten quotes goes to a human underwriter rather than being quoted automatically. Your head of digital distribution is concerned the referral system will disproportionately affect customers in certain demographics and breach Consumer Duty obligations. You need to (a) implement the flag, (b) show what it flags, and (c) demonstrate it is based on model uncertainty, not demographic proxies.

**What to do:**

1. Using the calibrated `pearson_weighted` predictor, generate 90% intervals for the test set. Compute relative interval width `(upper - lower) / point`. Set the threshold at the 90th percentile. Verify the flag rate is exactly 10%.

2. Build a summary table comparing the flagged and unflagged populations on: mean predicted pure premium, mean actual loss cost, mean age, mean vehicle group, mean vehicle age, and mean credit score. Which feature shows the largest difference between flagged and unflagged?

3. Compute the actual empirical coverage (fraction where actual loss cost falls within the 90% interval) for the flagged group and the unflagged group separately. Is coverage materially different between the groups? Should it be?

4. A 72-year-old driver in area A with NCD=5 and no convictions but driving vehicle group 49 has a very wide interval. A 23-year-old in area D with NCD=0 and 6 conviction points in vehicle group 25 has a narrower interval. Explain why, in plain English that you would use with the underwriting director.

5. **Extension (optional):** Compute the flag rate separately by area. Is it stable across areas (within ±3pp of 10%)? If not, which area has the highest referral rate and why?

**Start here:**

```python
# Exercise 3: Underwriting referral flag
# Use the main conformal predictor (calibrated on full calibration set)
intervals_90 = cp.predict_interval(X_test, alpha=0.10)
point        = intervals_90["point"].to_numpy()
lower        = intervals_90["lower"].to_numpy()
upper        = intervals_90["upper"].to_numpy()

print(f"Test set size: {len(X_test):,}")
print(f"Mean point estimate: £{point.mean():.2f}")
print(f"Mean interval width: £{(upper - lower).mean():.2f}")
print("\nNow complete Tasks 1-4.")
```

<details>
<summary>Hint for Task 3: should coverage differ between flagged and unflagged groups?</summary>

The flag is based on relative interval width, which is a proxy for the model's uncertainty. Wide-interval policies are flagged. The coverage guarantee is **marginal** - it applies to all observations combined, not to any specific subset.

In practice, coverage should be similar between flagged and unflagged groups because the `pearson_weighted` score is designed to equalise coverage across risk levels. If coverage is substantially lower for the flagged group (e.g. 80% vs 92%), it suggests the Pearson score has not fully corrected for the heteroscedasticity in the thin-cell risks that drive the flags.

</details>

<details>
<summary>Hint for Task 4: thin cells vs common high-risk profiles</summary>

A young driver (23) with 6 conviction points in vehicle group 25 is in a cell that appears many times in training data. The training set has many such drivers. The model has seen enough examples of that combination to make a reliable prediction. The interval is narrower because the calibration scores for similar risks cluster tightly around a predictable residual.

A 72-year-old in vehicle group 49 may appear only a handful of times in the entire training dataset. The model's prediction for this risk is extrapolation from similar-but-not-identical risks. There is genuine uncertainty about whether the model has correctly learnt the effect of that specific combination. The calibration scores for similar risks have high variance, and the interval reflects that.

Model uncertainty is about training data density in the relevant region of feature space. It is independent of the expected risk level.

</details>

<details>
<summary>Solution - Exercise 3</summary>

```python
import numpy as np

# Task 1: Relative width flag
rel_width        = (upper - lower) / np.clip(point, 1e-6, None)
threshold_10pct  = np.quantile(rel_width, 0.90)
flag             = rel_width > threshold_10pct

print(f"Threshold (90th percentile of relative width): {threshold_10pct:.4f}")
print(f"Flag rate: {flag.mean():.1%} (should be 10.0%)")

# Task 2: Profile comparison
X_test_arr = X_test.reset_index(drop=True)
y_test_arr = y_test.to_numpy()

for flag_val, label in [(True, "FLAGGED"), (False, "Not flagged")]:
    mask = flag == flag_val
    sub  = X_test_arr[mask]
    print(f"\n{label} ({mask.sum():,} policies):")
    print(f"  Mean point estimate:  £{point[mask].mean():.2f}")
    print(f"  Mean actual loss cost:  £{y_test_arr[mask].mean():.2f}")
    print(f"  Mean age:               {sub['age'].mean():.1f} years")
    print(f"  Mean vehicle group:     {sub['vehicle_group'].mean():.1f}")
    print(f"  Mean vehicle age:       {sub['vehicle_age'].mean():.1f}")
    print(f"  Mean credit score:      {sub['credit_score'].mean():.1f}")
    print(f"  Mean relative width:    {rel_width[mask].mean():.3f}")

# Task 3: Coverage by flag group
print()
for flag_val, label in [(True, "Flagged"), (False, "Not flagged")]:
    mask    = flag == flag_val
    covered = ((y_test_arr[mask] >= lower[mask]) & (y_test_arr[mask] <= upper[mask]))
    print(f"Coverage ({label}): {covered.mean():.3f} (target: 0.90)")

print("""
Interpretation: both groups should achieve similar coverage (~90%).
The flag is not identifying risks where coverage will fail - it is identifying
risks where the model is uncertain. The Pearson score should equalise coverage
across risk levels, so whether you are flagged or not should not materially
affect whether your actual outcome falls inside the interval.

If coverage is substantially lower for the flagged group, it means the Pearson
score has not fully corrected the heteroscedasticity for those risks.
""")

# Task 4 explanation
print("""
Task 4: Why the 72-year-old in vehicle group 49 has a wider interval than
the 23-year-old with 6 conviction points in vehicle group 25:

Training data density is the driver of model uncertainty, not risk level.

The 23-year-old with 6 conviction points is a high-risk profile, but it is
a COMMON high-risk profile. The training set contains thousands of young drivers
with multiple conviction points in mid-range vehicle groups. The model has seen
enough examples to learn the risk accurately. The calibration scores for similar
observations cluster tightly - the model is reliably wrong by a predictable amount.
The interval is narrower because the model's residuals for that profile are consistent.

The 72-year-old in vehicle group 49 may appear only a handful of times in the
entire 60,000-row training set. The model must interpolate from elderly drivers
in lower vehicle groups and mid-age drivers in vehicle group 49 - neither of which
is exactly the same risk. The calibration scores for similar observations have high
variance - the model is wrong by different amounts each time. The interval is wider
because the model's errors in that region are inconsistent, reflecting genuine
uncertainty about the correct prediction.

High risk and high uncertainty are independent dimensions. Both can be true simultaneously,
but neither implies the other.
""")

# Extension: flag rate by area
print("Extension: Flag rate by area")
for area_val in sorted(X_test_arr["area"].unique()):
    mask      = X_test_arr["area"] == area_val
    area_flag = flag[mask.values]
    print(f"  Area {area_val}: flag rate {area_flag.mean():.1%} ({mask.sum():,} policies)")
```

**Expected output for Task 1:** flag rate should be exactly 10.0% by construction.

**Expected output for Task 3:** both groups should show coverage between 87-93%. If the flagged group shows coverage below 85%, revisit the `pearson_weighted` score and consider whether a `deviance` score performs better.

</details>

---

## Exercise 4: Minimum premium floors from conformal upper bounds

**Objective.** Build a risk-specific minimum premium floor using the conformal upper bound, compare it to the conventional approach, and produce the Consumer Duty evidence that the new approach does not systematically overcharge any customer segment.

**Business context.** Your minimum premium policy currently reads: "Minimum premium = 1.3x technical premium, subject to a floor of £250." You are proposing a conformal-based floor. Your Consumer Duty team needs (a) evidence that the new floor does not overcharge low-volatility risks, and (b) evidence that it reflects genuine risk-based uncertainty for high-volatility risks. The FCA could ask about this at your next supervisory interview.

**What to do:**

1. Compute three floor candidates for each test observation:
   - `floor_conventional`: `max(1.3 × point, 250)`
   - `floor_conformal_95`: upper bound of the 95% interval
   - `floor_practical`: `max(1.5 × point, upper bound of the 80% interval)`

   Report median, mean, and 95th percentile for each approach.

2. Find policies where `floor_conformal_95 > floor_conventional`. What is the typical risk profile of these policies? Compute mean driver age, mean vehicle group, percentage with conviction points, and mean NCD years.

3. Find policies where `floor_conformal_95 < floor_conventional`. These are the policies where the conventional flat multiplier overcharges relative to the conformal floor. What is their typical profile? Calculate the mean percentage overcharge: `(floor_conventional - floor_conformal_95) / floor_conformal_95`.

4. Plot a scatter of `floor_conventional` vs `floor_conformal_95` for a random sample of 500 test policies. Points above the diagonal are cases where the conformal floor is higher. Points below are cases where the conventional floor is higher. The scatter should show a clear pattern.

5. Draft a four-sentence Consumer Duty argument explaining why the conformal floor is consistent with fair value obligations, covering both groups identified in Tasks 2 and 3.

**Start here:**

```python
# Exercise 4: Minimum premium floors
# Generate intervals at different alpha levels
intervals_95 = cp.predict_interval(X_test, alpha=0.05)
intervals_80 = cp.predict_interval(X_test, alpha=0.20)

point_est = cp.predict_interval(X_test, alpha=0.10)["point"].to_numpy()
upper_95  = intervals_95["upper"].to_numpy()
upper_80  = intervals_80["upper"].to_numpy()

print(f"Test set size: {len(X_test):,} policies")
print(f"Mean point estimate: £{point_est.mean():.2f}")
print(f"Mean 95% upper bound: £{upper_95.mean():.2f}")
print(f"Mean 80% upper bound: £{upper_80.mean():.2f}")
print("\nNow compute the three floor candidates.")
```

<details>
<summary>Hint for Task 4: making a diagonal scatter plot</summary>

```python
sample_idx = np.random.choice(len(floor_conventional), size=500, replace=False)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(floor_conventional[sample_idx], floor_conformal_95[sample_idx],
           alpha=0.4, s=15, c="steelblue")
# Add a 45-degree line (y=x) to show where the two approaches agree
lims = [0, max(floor_conventional.max(), floor_conformal_95.max())]
ax.plot(lims, lims, "k--", linewidth=1, label="Equal floors")
ax.set_xlabel("Conventional floor (1.3x)")
ax.set_ylabel("Conformal 95% floor")
ax.set_title("Minimum premium floor comparison")
ax.legend()
plt.tight_layout()
plt.show()
```

Points above the diagonal: conformal floor > conventional (volatile risks, conventional undercharges).
Points below the diagonal: conventional floor > conformal (stable risks, conventional overcharges).

</details>

<details>
<summary>Solution - Exercise 4</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Three floor approaches
floor_conventional = np.maximum(1.3 * point_est, 250)
floor_conformal_95 = upper_95
floor_practical    = np.maximum(1.5 * point_est, upper_80)

print(f"{'Approach':<38} {'Median':>10} {'Mean':>10} {'95th pctile':>14}")
print("-" * 76)
for label, floor in [
    ("Conventional (1.3x, floor £250)",     floor_conventional),
    ("Conformal 95% upper bound",           floor_conformal_95),
    ("Practical (1.5x vs 80% upper)",       floor_practical),
]:
    print(f"{label:<38} £{np.median(floor):>9.2f} £{np.mean(floor):>9.2f} £{np.quantile(floor, 0.95):>13.2f}")

# Task 2: Where conformal floor is higher (conventional undercharges)
higher_idx = floor_conformal_95 > floor_conventional
print(f"\nTask 2: Conformal floor > conventional: {higher_idx.sum():,} policies ({higher_idx.mean():.1%})")
X_test_arr = X_test.reset_index(drop=True)
sub_hi = X_test_arr[higher_idx]
sub_all = X_test_arr
print(f"  Mean age:             {sub_hi['age'].mean():.1f} vs portfolio {sub_all['age'].mean():.1f}")
print(f"  Mean vehicle group:   {sub_hi['vehicle_group'].mean():.1f} vs portfolio {sub_all['vehicle_group'].mean():.1f}")
print(f"  Mean vehicle age:     {sub_hi['vehicle_age'].mean():.1f} vs portfolio {sub_all['vehicle_age'].mean():.1f}")
print(f"  Mean credit score:    {sub_hi['credit_score'].mean():.1f} vs portfolio {sub_all['credit_score'].mean():.1f}")

# Task 3: Where conformal floor is lower (conventional overcharges)
lower_idx = floor_conformal_95 < floor_conventional
print(f"\nTask 3: Conventional floor > conformal: {lower_idx.sum():,} policies ({lower_idx.mean():.1%})")
sub_lo = X_test_arr[lower_idx]
print(f"  Mean age:             {sub_lo['age'].mean():.1f}")
print(f"  Mean vehicle group:   {sub_lo['vehicle_group'].mean():.1f}")
print(f"  Mean vehicle age:     {sub_lo['vehicle_age'].mean():.1f}")
print(f"  Mean credit score:    {sub_lo['credit_score'].mean():.1f}")

overcharge_pct = (floor_conventional[lower_idx] - floor_conformal_95[lower_idx]) / floor_conformal_95[lower_idx]
print(f"  Mean % conventional overcharge: {overcharge_pct.mean() * 100:.1f}%")
print(f"  Median % conventional overcharge: {np.median(overcharge_pct) * 100:.1f}%")

# Task 4: Scatter plot
sample_idx = np.random.default_rng(42).choice(len(floor_conventional), size=500, replace=False)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(floor_conventional[sample_idx], floor_conformal_95[sample_idx],
           alpha=0.4, s=15, c="steelblue", label="Policies")
lim_max = max(np.quantile(floor_conventional, 0.99), np.quantile(floor_conformal_95, 0.99))
ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="Equal floors")
ax.set_xlabel("Conventional floor (1.3x, floor £250)")
ax.set_ylabel("Conformal 95% floor")
ax.set_title("Minimum premium floor comparison (500-policy sample)")
ax.legend()
plt.tight_layout()
plt.show()

# Task 5: Consumer Duty argument
print("""
Task 5 - Consumer Duty argument:

A conformal minimum premium floor is risk-specific: it is set at the 95th percentile
of predicted outcomes for each individual policy, validated to achieve 95% coverage
across all risk deciles on recent historical business. This means the floor reflects
the actual distribution of outcomes for each risk, rather than an arbitrary multiplier
applied uniformly.

For customers in Group 2 (where the conformal floor is lower than the conventional floor),
the conventional 1.3x approach has been systematically overcharging relative to the
principled uncertainty-based floor: the data shows that 95% of outcomes for those risks
fall below the conformal floor, which is well below the conventional floor. Switching to
the conformal approach removes this overcharging and improves fair value.

For customers in Group 1 (where the conformal floor is higher), the higher floor reflects
genuine uncertainty about loss outcomes for high-volatility risk profiles. This is not
discrimination - it is an actuarially grounded response to the fact that outcomes for
those risks are more variable, and the FCA's expectation is that pricing should reflect
risk appropriately.

The evidence of fair value is the coverage-by-decile diagnostic showing 95% coverage
uniformly across risk deciles: this demonstrates the floor is calibrated on data, not
set by judgment, and that it is consistently applied without systematic advantage or
disadvantage to any specific risk group.
""")
```

</details>

---

## Exercise 5: Detecting and responding to coverage drift

**Objective.** Build a coverage monitoring function that detects when conformal intervals are no longer achieving stated coverage on recent business, and demonstrate that recalibration (without retraining) restores coverage.

**Business context.** Your conformal predictor was calibrated on 2022-2023 business. It is now Q3 2025. Claim frequency has risen 12% due to a combination of weather events and economic pressures on repair costs. Your reserving team is about to use the intervals for the year-end reserve range. Before they do, you need to check whether the intervals are still valid.

**What to do:**

1. Simulate distribution drift by generating a new test cohort with 20% higher losses than the original test set. Compute 90% intervals using the predictor calibrated on the original data. What is the marginal coverage? What is the coverage by decile? Is the current calibration still valid?

2. Plot the non-conformity score distribution for the original calibration set (`cp.calibration_scores_`) and for the drifted test data (compute the scores manually: `|y_drifted - ŷ| / ŷ^0.75`). How has the distribution shifted?

3. Write a function `check_coverage_drift(cp, X_new, y_new, alpha=0.10, alert_threshold=0.05)` that:
   - Generates intervals on `X_new, y_new`
   - Computes marginal coverage
   - Returns a dictionary with keys `coverage`, `target`, `gap`, and `needs_recalibration` (a boolean)
   - If the gap exceeds `alert_threshold`, prints a warning: "Coverage drift detected: observed=X.XXX, target=X.XX, gap=X.XXX. Recalibrate immediately."

4. Recalibrate the predictor using 2,000 observations from the drifted cohort (simulating using recent live data). Recheck coverage on the remaining drifted observations. Does recalibration restore coverage? How long does it take compared to retraining the base model?

5. **Extension:** Recalibration works when the base model's rankings are still correct but the error scale has shifted. To test whether rankings have degraded, compute the Gini coefficient (or Spearman rank correlation between predicted and actual loss cost) on the original test set and on the drifted cohort. If the Gini falls by more than 5pp, it is a signal that the base model needs retraining, not just recalibration.

**Start here:**

```python
# Exercise 5: Coverage drift simulation
# Simulate 2025 conditions: 20% loss inflation across the board
rng_drift = np.random.default_rng(seed=77)
n_drift   = 10_000

# Generate a new cohort with the same feature distribution
# Generate drifted cohort using load_motor() scaled by 20%
age_d         = rng_drift.integers(17, 86, size=n_drift)
vehicle_grp_d = rng_drift.integers(1, 51, size=n_drift)
exp_d         = np.clip(rng_drift.beta(8, 2, size=n_drift), 0.05, 1.0)

# Simplified DGP consistent with load_motor() structure
log_mu_d = (
    -3.10
    + 0.010 * (vehicle_grp_d - 25)
    + np.where(age_d < 25, 0.55, np.where(age_d > 70, 0.20, 0.0))
)
sev_log_mu_d = 7.80 + 0.015 * (vehicle_grp_d - 25)
claim_count_d = rng_drift.poisson(np.exp(log_mu_d) * exp_d)
incurred_d    = np.where(
    claim_count_d > 0,
    rng_drift.gamma(3.0, np.exp(sev_log_mu_d) / 3.0, size=n_drift) * claim_count_d * 1.20,  # 20% drift
    0.0,
)
pure_prem_d = incurred_d / exp_d

import pandas as pd
# X_drift uses same columns as X_COLS defined in the setup cell
X_drift = pd.DataFrame({
    "age":           age_d.astype(np.int32),
    "vehicle_age":   rng_drift.integers(0, 16, size=n_drift).astype(np.int32),
    "vehicle_group": vehicle_grp_d.astype(np.int32),
    "region":        rng_drift.choice(["North", "Midlands", "SouthEast", "London", "SouthWest"], size=n_drift),
    "credit_score":  rng_drift.integers(300, 850, size=n_drift).astype(np.int32),
})
y_drift = pd.Series(pure_prem_d)

print(f"Drifted cohort: {n_drift:,} policies")
print(f"Mean actual loss (drifted): £{pure_prem_d.mean():.2f}")
print(f"Mean actual loss (original test): £{y_test.mean():.2f}")
print(f"Ratio (drift / original): {pure_prem_d.mean() / y_test.mean():.3f}")
```

<details>
<summary>Hint for Task 2: computing non-conformity scores manually</summary>

The Pearson-weighted non-conformity score is:

```python
score = |y - ŷ| / ŷ^(p/2)
```

For Tweedie p=1.5:

```python
preds_drift = model.predict(Pool(X_drift, cat_features=CAT_FEATURES))
scores_drift = np.abs(y_drift.values - preds_drift) / (np.clip(preds_drift, 1e-6, None) ** 0.75)
```

Compare to `cp.calibration_scores_` (sorted array of calibration scores). The distribution of test scores will have shifted right: the 90th percentile of drift scores will be above the 90th percentile of calibration scores, which is why coverage falls.

</details>

<details>
<summary>Hint for Task 4: recalibrating on drifted data</summary>

Use the first 2,000 rows of the drifted cohort for recalibration:

```python
X_recal = X_drift.iloc[:2_000]
y_recal = y_drift.iloc[:2_000]
cp.calibrate(X_recal, y_recal)
```

This overwrites the stored calibration scores with scores computed on the drifted data. The new 90th percentile calibration score is higher, producing wider intervals that achieve the target coverage on the drifted distribution.

Test coverage on the remaining rows: `X_drift.iloc[2_000:]` and `y_drift.iloc[2_000:]`.

</details>

<details>
<summary>Solution - Exercise 5</summary>

```python
import numpy as np
import time
import matplotlib.pyplot as plt
from catboost import Pool

# Task 1: Coverage with original calibration on drifted data
ivs_drift   = cp.predict_interval(X_drift, alpha=0.10)
diag_drift  = cp.coverage_by_decile(X_drift, y_drift, alpha=0.10)

y_d_arr = y_drift.to_numpy()
covered_drift = (
    (y_d_arr >= ivs_drift["lower"].to_numpy()) &
    (y_d_arr <= ivs_drift["upper"].to_numpy())
)
print(f"Marginal coverage on drifted data: {covered_drift.mean():.3f} (target: 0.90)")
print("\nCoverage by decile (drifted data, original calibration):")
print(diag_drift)

if covered_drift.mean() < 0.85:
    print("\nWARNING: Marginal coverage has fallen below 85%. Recalibrate immediately.")

# Task 2: Non-conformity score distribution shift
preds_drift  = model.predict(Pool(X_drift, cat_features=CAT_FEATURES))
scores_drift = np.abs(y_d_arr - preds_drift) / (np.clip(preds_drift, 1e-6, None) ** 0.75)
cal_scores   = cp.calibration_scores_

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(cal_scores,   bins=60, alpha=0.5, density=True, label="Calibration scores (original)", color="steelblue")
ax.hist(scores_drift, bins=60, alpha=0.5, density=True, label="Non-conformity scores (drifted 2025)", color="firebrick")
q90_cal   = np.quantile(cal_scores, 0.90)
q90_drift = np.quantile(scores_drift, 0.90)
ax.axvline(q90_cal,   color="steelblue",  linestyle="--", linewidth=1.5, label=f"Calibration 90th pctile ({q90_cal:.3f})")
ax.axvline(q90_drift, color="firebrick",  linestyle="--", linewidth=1.5, label=f"Drift 90th pctile ({q90_drift:.3f})")
ax.set_xlabel("Non-conformity score (Pearson residual)")
ax.set_ylabel("Density")
ax.set_title("Score distribution shift: calibration (2023) vs drifted test (2025)")
ax.legend()
plt.tight_layout()
plt.show()

print(f"\nCalibration 90th percentile score: {q90_cal:.3f}")
print(f"Drifted test 90th percentile score: {q90_drift:.3f}")
print(f"Drift ratio: {q90_drift / q90_cal:.2f}x")
print("The test scores have shifted right - the model is more wrong on drifted data than expected.")

# Task 3: Coverage drift detection function
def check_coverage_drift(
    cp: InsuranceConformalPredictor,
    X_new,
    y_new,
    alpha: float = 0.10,
    alert_threshold: float = 0.05,
) -> dict:
    """
    Check whether conformal intervals achieve stated coverage on new data.

    Parameters
    ----------
    cp               : calibrated InsuranceConformalPredictor
    X_new            : features (pandas DataFrame)
    y_new            : outcomes (pandas Series or array)
    alpha            : target miscoverage rate (1 - alpha = target coverage)
    alert_threshold  : warn if gap > alert_threshold

    Returns
    -------
    dict with coverage, target, gap, needs_recalibration
    """
    target    = 1.0 - alpha
    ivs       = cp.predict_interval(X_new, alpha=alpha)
    y_arr     = y_new.to_numpy() if hasattr(y_new, "to_numpy") else np.asarray(y_new)
    covered   = (y_arr >= ivs["lower"].to_numpy()) & (y_arr <= ivs["upper"].to_numpy())
    coverage  = float(covered.mean())
    gap       = target - coverage

    result = {
        "coverage":            round(coverage, 4),
        "target":              target,
        "gap":                 round(gap, 4),
        "needs_recalibration": gap > alert_threshold,
    }

    if result["needs_recalibration"]:
        print(f"WARNING: Coverage drift detected: observed={coverage:.3f}, "
              f"target={target:.2f}, gap={gap:.3f} > threshold={alert_threshold:.2f}. "
              f"Recalibrate immediately.")
    else:
        print(f"Coverage OK: observed={coverage:.3f}, target={target:.2f}, gap={gap:.3f}.")

    return result

# Test the function
print("\nRunning drift check on drifted data:")
result = check_coverage_drift(cp, X_drift, y_drift, alpha=0.10, alert_threshold=0.05)
print(result)

# Task 4: Recalibrate on recent drifted data
X_recal   = X_drift.iloc[:2_000]
y_recal   = y_drift.iloc[:2_000]
X_test_d  = X_drift.iloc[2_000:]
y_test_d  = y_drift.iloc[2_000:]

print(f"\nRecalibrating on {len(X_recal):,} recent observations from drifted cohort...")
t0 = time.time()
cp.calibrate(X_recal, y_recal)
recal_time = time.time() - t0
print(f"Recalibration complete in {recal_time:.2f} seconds")

print("\nCoverage after recalibration:")
result_after = check_coverage_drift(cp, X_test_d, y_test_d, alpha=0.10, alert_threshold=0.05)
print(result_after)

diag_recal = cp.coverage_by_decile(X_test_d, y_test_d, alpha=0.10)
print("\nCoverage by decile (after recalibration):")
print(diag_recal)

print(f"""
Recalibration took {recal_time:.2f} seconds.
Retraining the full CatBoost model on 60,000 rows would take approximately 15-25 minutes.

Recalibration works here because the base model's predictions are still directionally
correct - it still ranks risks in approximately the right order. The drift is in the
scale of errors (losses are 20% higher), not in the ranking. Recalibration adjusts the
quantile threshold to reflect the new error scale.

If the Gini coefficient had fallen substantially (see Extension), it would signal that
the model's rankings have degraded. In that case, recalibration would restore marginal
coverage but coverage-by-decile would remain poor, indicating the base model itself
needs retraining.
""")

# Extension: Gini degradation check
def gini(y_true, y_pred):
    """Normalised Gini coefficient: discrimination measure for regression."""
    n     = len(y_true)
    order = np.argsort(y_pred)
    y_s   = y_true[order]
    lorenz = y_s.cumsum() / y_s.sum()
    return 1 - 2 * lorenz.mean()

# Restore original calibration for fair comparison
cp.calibrate(X_cal, y_cal)

preds_orig  = model.predict(test_pool)
preds_drift_arr = model.predict(Pool(X_drift, cat_features=CAT_FEATURES))

gini_orig  = gini(y_test.to_numpy(), preds_orig)
gini_drift = gini(y_drift.to_numpy(), preds_drift_arr)

print(f"Extension: Gini comparison")
print(f"  Original test set Gini: {gini_orig:.3f}")
print(f"  Drifted cohort Gini:    {gini_drift:.3f}")
print(f"  Change:                 {gini_drift - gini_orig:+.3f}")
if abs(gini_drift - gini_orig) < 0.05:
    print("  Gini is stable. Recalibration is sufficient. No need to retrain the base model.")
else:
    print("  Gini has fallen by more than 5pp. The model's risk ranking has degraded.")
    print("  Recalibration will restore marginal coverage but coverage-by-decile may remain poor.")
    print("  Retrain the base model.")
```

**Expected outcome for Task 4:** coverage after recalibration should return to approximately 90% overall and should pass the coverage-by-decile check. Recalibration time should be under 5 seconds. The Gini in the Extension should be approximately stable (the drift is in scale, not ranking), confirming that recalibration is sufficient without retraining.

**What to note when presenting results:** if you run the drift check quarterly and coverage falls below 85% before the next scheduled calibration, the `check_coverage_drift` function provides a clear trigger for unscheduled recalibration. The function should be integrated into your model monitoring pipeline and run on each new batch of settled claims.

</details>

---

## Summary: what you should now be able to do

After completing these exercises, you should be able to:

1. **Explain why raw residual intervals fail for insurance** and choose the correct non-conformity score (Exercise 1)
2. **Recommend a minimum calibration set size** based on bootstrap precision analysis and justify it to your chief actuary (Exercise 2)
3. **Build and characterise an underwriting referral flag** based on model uncertainty, and explain the difference between uncertainty and risk level to a non-technical audience (Exercise 3)
4. **Replace a flat minimum premium multiplier** with a risk-specific conformal floor and produce the Consumer Duty evidence that the new approach is fair (Exercise 4)
5. **Monitor coverage over time** and distinguish between distribution drift (fix with recalibration) and model degradation (fix with retraining) (Exercise 5)

The eight limitations in the tutorial are the governance deliverable. If your pricing committee or the FCA asks about the mathematical basis of these intervals, the answer is: marginal coverage guarantee with finite-sample correction `1/(n+1)`, conditional on exchangeability between calibration and test data, validated on recent business with coverage-by-decile, with documented limitations as listed.
