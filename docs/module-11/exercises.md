# Module 11 Exercises: Model Monitoring and Drift Detection

Seven exercises. They are designed to be worked through in order — the setup from Exercise 1 is reused throughout. Exercises 6 and 7 can be done independently once Exercise 5 is complete.

Each exercise takes 20–40 minutes. Solutions are hidden in `<details>` blocks. Read the question before looking.

---

## Before you start

These exercises use `insurance-datasets` and `insurance-monitoring`. If you are starting in a fresh notebook:

```python
%pip install insurance-datasets insurance-monitoring catboost polars scikit-learn --quiet
```

```python
dbutils.library.restartPython()
```

Then run the shared setup cell below. All exercises assume this setup has been run.

**Shared setup — run this first:**

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from insurance_datasets import load_motor
from insurance_monitoring import (
    PSICalculator,
    CSICalculator,
    AERatio,
    GiniDrift,
    MonitoringReport,
)

# ---------------------------------------------------------------------------
# Load data and split into reference / current windows
# ---------------------------------------------------------------------------
df = load_motor()

# Reference period: policies with accident_year <= 2022
# Current period:   policies with accident_year == 2023
df_reference = df.filter(pl.col("accident_year") <= 2022)
df_current   = df.filter(pl.col("accident_year") == 2023)

print(f"Reference: {df_reference.shape[0]:,} rows")
print(f"Current:   {df_current.shape[0]:,} rows")
print()
print("Reference accident years:", df_reference["accident_year"].unique().sort().to_list())
print("Current accident years:  ", df_current["accident_year"].unique().sort().to_list())

# ---------------------------------------------------------------------------
# Train a CatBoost frequency model on the reference period
# ---------------------------------------------------------------------------
FEATURES     = ["driver_age", "vehicle_age", "vehicle_group", "region",
                 "ncb_years", "annual_mileage"]
CAT_FEATURES = ["vehicle_group", "region"]

rng      = np.random.default_rng(seed=42)
n_ref    = df_reference.shape[0]
is_train = rng.random(n_ref) < 0.8

df_train = df_reference.filter(pl.Series(is_train))
df_val   = df_reference.filter(pl.Series(~is_train))

def make_pool(df: pl.DataFrame, with_label: bool = True):
    X = df.select(FEATURES).to_pandas()
    if with_label:
        y = df["claim_count"].to_numpy().astype(float)
        w = df["exposure"].to_numpy()
        return Pool(X, label=y, baseline=np.log(np.clip(w, 1e-6, None)),
                    cat_features=CAT_FEATURES)
    return Pool(X, cat_features=CAT_FEATURES)

train_pool = make_pool(df_train)
val_pool   = make_pool(df_val)

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    depth=5,
    learning_rate=0.05,
    random_seed=42,
    verbose=0,
)
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)
print(f"Model trained. Best iteration: {model.best_iteration_}")

# ---------------------------------------------------------------------------
# Generate predictions for both windows
# ---------------------------------------------------------------------------
pred_ref = model.predict(make_pool(df_reference, with_label=False))
pred_cur = model.predict(make_pool(df_current,   with_label=False))

exposure_ref = df_reference["exposure"].to_numpy()
exposure_cur = df_current["exposure"].to_numpy()

expected_ref = pred_ref * exposure_ref
expected_cur = pred_cur * exposure_cur

print(f"\nReference predictions: mean={pred_ref.mean():.4f}")
print(f"Current predictions:   mean={pred_cur.mean():.4f}")
print("\nSetup complete.")
```

---

## Exercise 1: PSI — calculation, binning, and what the number means

**References:** Tutorial Part 4.

**What this exercise covers:** PSI is a single number but the information is in the bins. This exercise takes you from first principles to the `PSICalculator` API, then asks you to identify which part of the risk distribution is driving the shift.

### Task 1: Implement PSI from scratch

Write a function `psi_from_scratch` that computes PSI without using `insurance_monitoring`. Use the reference distribution to define 10 equal-frequency bins, then map the current distribution onto those bins.

```python
def psi_from_scratch(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> tuple[float, pl.DataFrame]:
    """
    Compute PSI using reference-quantile bins.

    Returns
    -------
    psi   : float
        Scalar PSI value.
    bins_df : pl.DataFrame
        One row per bin with columns:
        lower, upper, ref_pct, cur_pct, contribution
    """
    # Your implementation here.
    pass


psi_manual, bins_df = psi_from_scratch(pred_ref, pred_cur)
print(f"Manual PSI: {psi_manual:.4f}")
print(bins_df)
```

Requirements:
- Bins must be defined from the **reference** distribution percentiles (0%, 10%, 20%, ..., 100%). Use the 0th and 100th percentiles as the outer edges with a small buffer so that all current observations fall within a bin.
- Both `ref_pct` and `cur_pct` must sum to 1.0 (within floating-point tolerance).
- Use `eps` to avoid log(0): replace any zero proportion with `eps` before computing `ln(cur/ref)`.
- The function must return the scalar PSI **and** the bin-level breakdown.

### Task 2: Verify against the library

Run `PSICalculator` on the same data and compare.

```python
psi_calc   = PSICalculator(n_bins=10)
psi_result = psi_calc.calculate(
    reference=pred_ref,
    current=pred_cur,
    exposure_ref=exposure_ref,
    exposure_cur=exposure_cur,
)
print(f"Library PSI (exposure-weighted): {psi_result.psi:.4f}")
print(f"Manual PSI (unweighted):         {psi_manual:.4f}")
print(f"Traffic light: {psi_result.traffic_light}")
```

**Question 2a:** Why does the library result differ from your manual result? The library accepts `exposure_ref` and `exposure_cur` arguments. Write one paragraph explaining what the exposure-weighted PSI measures and why unweighted PSI can mislead on insurance data. Give a concrete example of the type of seasonal portfolio composition that would cause unweighted PSI to flag a false positive.

**Question 2b:** The thresholds (0.10 / 0.20) originate from credit scoring in the 1980s. Name one structural difference between a credit scoring application and a UK motor frequency model that might mean these thresholds are calibrated poorly for insurance. Should the threshold be higher or lower for motor, and why?

### Task 3: Diagnose where the shift is concentrated

Plot the per-bin PSI contributions from your `bins_df`. Colour each bar green (< 0.02), amber (0.02–0.05), or red (> 0.05).

```python
# Your plot here
# Title should include the total PSI value
# x-axis: bin range (e.g. "0.04-0.06")
# y-axis: contribution to PSI
# Horizontal reference lines at 0.02 and 0.05
```

**Question 3:** Is the shift concentrated in the high-risk tail (top two bins), the low-risk body (bottom five bins), or spread evenly? From a pricing perspective, which concentration pattern is more concerning, and why?

<details>
<summary>Solution — Exercise 1</summary>

```python
# Task 1: PSI from scratch
def psi_from_scratch(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> tuple[float, pl.DataFrame]:
    quantiles  = np.linspace(0, 100, n_bins + 1)
    edges      = np.percentile(reference, quantiles)
    # Extend edges slightly so all points fall inside
    edges[0]  -= 1e-8
    edges[-1] += 1e-8

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current,   bins=edges)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Smooth zeros
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    contributions = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi = contributions.sum()

    bins_df = pl.DataFrame({
        "lower":        edges[:-1].tolist(),
        "upper":        edges[1:].tolist(),
        "ref_pct":      ref_pct.tolist(),
        "cur_pct":      cur_pct.tolist(),
        "contribution": contributions.tolist(),
    })
    return float(psi), bins_df


psi_manual, bins_df = psi_from_scratch(pred_ref, pred_cur)
print(f"Manual PSI: {psi_manual:.4f}")
print(bins_df)

# Task 3: Per-bin plot
fig, ax = plt.subplots(figsize=(12, 5))

labels = [
    f"{row['lower']:.3f}-{row['upper']:.3f}"
    for row in bins_df.iter_rows(named=True)
]
contribs = bins_df["contribution"].to_list()
colours  = ["green" if c < 0.02 else "orange" if c < 0.05 else "red"
            for c in contribs]

ax.bar(range(len(contribs)), contribs, color=colours)
ax.axhline(0.02, color="orange", linestyle="--", linewidth=1.2, label="Amber (0.02)")
ax.axhline(0.05, color="red",    linestyle="--", linewidth=1.2, label="Red (0.05)")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Predicted frequency bin")
ax.set_ylabel("Contribution to PSI")
ax.set_title(f"PSI bin contributions  (total PSI = {psi_manual:.4f})")
ax.legend()
plt.tight_layout()
plt.show()
```

**Answer to 2a:** The exposure-weighted PSI weights each observation by its exposure before computing bucket proportions. Unweighted PSI treats a policy with 0.08 years of cover identically to one with a full year. A December renewal spike, or any period when short-term policies make up a larger share of new business, will distort the unweighted current distribution towards the risk profile of that short-term cohort — flagging a population shift that is entirely a coverage-duration artefact rather than a genuine change in the risk mix.

**Answer to 2b:** Credit scoring models are validated on application populations where each applicant is equally weighted (one applicant = one score). Motor pricing models score policies weighted by exposure, so a policy renewing mid-year contributes fractionally. The effective sample size in PSI terms is in earned exposures, not policy counts. Because insurance portfolios contain many short-term policies, unweighted PSI overstates sample sizes and therefore under-states the width of the "normal" noise band. The 0.10 threshold probably needs to be somewhat higher for motor — perhaps 0.12–0.15 — because more of what looks like drift is exposure-mix noise.

</details>

---

## Exercise 2: CSI — feature-level diagnosis

**References:** Tutorial Part 5.

**What this exercise covers:** PSI tells you the score has shifted. CSI tells you which features shifted and by how much. This exercise makes you run CSI on all model features, rank them, and use the results to form a hypothesis about what is driving any score-level shift.

### Task 1: Run CSI on all model features

```python
csi_calc    = CSICalculator(n_bins=10)
csi_results = {}

for feature in FEATURES:
    ref_values = df_reference[feature].to_numpy()
    cur_values = df_current[feature].to_numpy()

    result = csi_calc.calculate(
        feature_name=feature,
        reference=ref_values,
        current=cur_values,
    )
    csi_results[feature] = result

# Print ranked table — most-shifted feature first
print(f"{'Feature':<25} {'CSI':>8}  {'Status'}")
print("-" * 45)
for feature, result in sorted(csi_results.items(), key=lambda x: x[1].csi, reverse=True):
    flag = " <-- investigate" if result.csi > 0.20 else ""
    print(f"{feature:<25} {result.csi:>8.4f}  {result.traffic_light}{flag}")
```

### Task 2: Plot distribution comparisons for flagged features

For any feature with CSI > 0.10, produce a side-by-side histogram of the reference and current distributions. If no features exceed 0.10, use the top two features by CSI value.

```python
# Your plot here.
# One subplot per flagged feature.
# Reference: steelblue. Current: tomato.
# Title each subplot with the feature name and CSI value.
```

### Task 3: Categorical feature category check

`vehicle_group` and `region` are categorical. Check whether any categories are new in the current period (present in current, absent in reference) or have disappeared (present in reference, absent in current).

```python
for feature in CAT_FEATURES:
    ref_cats = set(df_reference[feature].unique().to_list())
    cur_cats = set(df_current[feature].unique().to_list())

    new_cats     = cur_cats - ref_cats
    missing_cats = ref_cats - cur_cats

    print(f"\n{feature}:")
    print(f"  New in current:      {sorted(new_cats) if new_cats else 'none'}")
    print(f"  Missing in current:  {sorted(missing_cats) if missing_cats else 'none'}")
```

**Question 3:** If a new `vehicle_group` category appears in current data, explain why `CSICalculator` does not raise an exception and instead produces a valid (if conservative) CSI. What smoothing technique handles the zero-proportion bin? When would you need to investigate the new category further, even if the CSI is below the amber threshold?

### Task 4: Connect CSI results to the score PSI

**Question 4:** Look at the features with the highest CSI values. Consider the direction of the shift (use your plots). Given that each feature has a non-zero importance in the CatBoost model, construct a one-paragraph narrative linking the feature shifts you observed to the score PSI from Exercise 1. Does the direction of the feature shifts predict the direction of the score distribution shift (i.e. does the current population look higher-risk or lower-risk than reference, and does that match the change in mean predicted frequency)?

<details>
<summary>Solution — Exercise 2</summary>

```python
# Task 2: Distribution plots for flagged features
flagged = [f for f, r in csi_results.items() if r.csi > 0.10]
if not flagged:
    flagged = sorted(csi_results, key=lambda f: csi_results[f].csi, reverse=True)[:2]

n = len(flagged)
fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
if n == 1:
    axes = [axes]

for ax, feature in zip(axes, flagged):
    ref_vals = df_reference[feature].to_numpy()
    cur_vals = df_current[feature].to_numpy()
    csi_val  = csi_results[feature].csi

    ax.hist(ref_vals, bins=30, alpha=0.6, label="Reference", color="steelblue", density=True)
    ax.hist(cur_vals, bins=30, alpha=0.6, label="Current",   color="tomato",    density=True)
    ax.set_title(f"{feature}  (CSI = {csi_val:.3f})")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.show()
```

**Answer to Question 3:** `CSICalculator` applies a small smoothing constant (1e-4) to any bin that has zero proportion in either the reference or current distribution before computing the log ratio. This prevents division by zero and undefined logarithms while keeping the contribution to CSI finite. A new category that was absent from reference has reference proportion ≈ 0 (after smoothing) and non-zero current proportion, producing a large but finite positive contribution.

Further investigation is warranted even when CSI is below amber if the new category maps to a high-risk cohort — for example, a new van hire segment that has never been modelled. The CSI may be green simply because the new category is small in volume, but the individual risk level could be far outside the model's training distribution.

</details>

---

## Exercise 3: A/E ratios with confidence intervals — what the CI is actually telling you

**References:** Tutorial Part 6.

**What this exercise covers:** A/E point estimates are essentially useless without a confidence interval. This exercise builds your intuition for when a deviation is signal versus noise, and forces you to compute segment-level A/E to find offsetting errors that the portfolio level hides.

### Task 1: Portfolio-level A/E

```python
ae_calc = AERatio()

actual_cur  = df_current["claim_count"].to_numpy().astype(float)

ae_portfolio = ae_calc.calculate(
    actual=actual_cur,
    expected=expected_cur,
    exposure=exposure_cur,
)

print(f"Portfolio A/E:    {ae_portfolio.ratio:.4f}")
print(f"95% CI:           [{ae_portfolio.ci_lower:.4f}, {ae_portfolio.ci_upper:.4f}]")
print(f"Actual claims:    {actual_cur.sum():.0f}")
print(f"Expected claims:  {expected_cur.sum():.1f}")
print(f"Traffic light:    {ae_portfolio.traffic_light}")
```

**Question 1:** The 95% CI formula under a Poisson assumption is approximately:

```
[A/E - 1.96 * sqrt(A) / E,   A/E + 1.96 * sqrt(A) / E]
```

where A = sum of actual claims and E = sum of expected claims. Compute this manually for the portfolio result above and verify it matches `ae_portfolio.ci_lower` and `ae_portfolio.ci_upper` (within rounding). Then explain in plain English what the confidence interval is telling a pricing analyst — not a statistician.

### Task 2: Segment-level A/E breakdown

Compute A/E separately for four driver age bands: 17–24, 25–39, 40–59, 60+. Print a table with A/E, 95% CI, actual claims, and expected claims for each band.

```python
age_bands = [
    (17, 25, "17-24"),
    (25, 40, "25-39"),
    (40, 60, "40-59"),
    (60, 120, "60+"),
]

df_cur_with_preds = df_current.with_columns(
    pl.Series("expected", expected_cur)
)

print(f"{'Band':<10} {'A/E':>8}  {'CI lower':>10}  {'CI upper':>10}  {'Actual':>8}  {'Expected':>10}  {'Status'}")
print("-" * 80)

for low, high, label in age_bands:
    seg = df_cur_with_preds.filter(
        (pl.col("driver_age") >= low) & (pl.col("driver_age") < high)
    )
    if seg.shape[0] == 0:
        continue

    result = ae_calc.calculate(
        actual=seg["claim_count"].to_numpy().astype(float),
        expected=seg["expected"].to_numpy(),
        exposure=seg["exposure"].to_numpy(),
    )
    print(
        f"{label:<10} {result.ratio:>8.4f}  {result.ci_lower:>10.4f}  "
        f"{result.ci_upper:>10.4f}  {seg['claim_count'].sum():>8.0f}  "
        f"{seg['expected'].sum():>10.1f}  {result.traffic_light}"
    )
```

**Question 2a:** Is there evidence of offsetting errors at segment level — i.e., does one age band show A/E > 1.0 with CI excluding 1.0, while another shows A/E < 1.0 with CI excluding 1.0? What would be the consequence if you applied a single portfolio-level recalibration factor in this case?

**Question 2b:** The 17–24 age band typically has the highest claim frequency but the fewest policies. What happens to the CI width for this band compared to the 40–59 band? What does this mean for your confidence in the 17–24 A/E estimate?

### Task 3: The A/E with 200 expected claims vs 20,000 expected claims

This task builds intuition for when an A/E is statistically meaningful.

```python
def ae_ci(actual_total: float, expected_total: float) -> tuple[float, float, float]:
    """Returns (A/E point estimate, CI lower, CI upper)."""
    ae = actual_total / expected_total
    margin = 1.96 * np.sqrt(actual_total) / expected_total
    return ae, ae - margin, ae + margin

# Scenario A: small portfolio
ae_a, lo_a, hi_a = ae_ci(actual_total=216, expected_total=200)
# Scenario B: large portfolio
ae_b, lo_b, hi_b = ae_ci(actual_total=21_600, expected_total=20_000)

print(f"Scenario A (E=200):   A/E={ae_a:.4f}  CI=[{lo_a:.4f}, {hi_a:.4f}]")
print(f"Scenario B (E=20000): A/E={ae_b:.4f}  CI=[{lo_b:.4f}, {hi_b:.4f}]")
```

**Question 3:** Both scenarios have an A/E of 1.08. In Scenario A the CI contains 1.0. In Scenario B it does not. Using the traffic light rules from Tutorial Part 8, what status does each scenario get, and why is that the correct behaviour? At what approximate expected claim count does an A/E of 1.08 become statistically distinguishable from 1.0 at the 95% level?

<details>
<summary>Solution — Exercise 3</summary>

```python
# Task 1: Manual CI verification
A = actual_cur.sum()
E = expected_cur.sum()
ae_point = A / E
margin   = 1.96 * np.sqrt(A) / E
ci_lo    = ae_point - margin
ci_hi    = ae_point + margin
print(f"Manual: A/E={ae_point:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
# Should match ae_portfolio.ci_lower and ae_portfolio.ci_upper
```

**Answer to Question 1:** The CI tells you the range of A/E values you would regard as consistent with the model being correctly calibrated given the amount of random variation you expect from a Poisson claims process. If 1.0 is inside the interval, you cannot rule out that the apparent over- or under-prediction is just bad luck from a finite number of claims. If 1.0 is outside the interval, you have 95% confidence that the model is systematically off, not just unlucky.

**Answer to Question 2a:** If one segment shows A/E of 1.20 (CI excludes 1.0) and another shows 0.85 (CI excludes 1.0), applying a portfolio recalibration factor of 1/1.04 = 0.96 would leave the under-predicted segment still under-predicted at 1.15, and make the over-predicted segment worse at 0.82. The portfolio average would look correct while both segments remain mis-priced.

**Answer to Question 3:** Scenario A is GREEN because the CI contains 1.0 — there is insufficient evidence to call the A/E elevated. Scenario B is AMBER (or RED depending on the absolute magnitude) because the CI excludes 1.0 — the same point estimate is statistically significant at this portfolio size. Solving for the break-even: 1.0 is excluded when `1.08 - 1.96 * sqrt(A) / E > 1.0`, i.e. `0.08 > 1.96 * sqrt(A) / E`. With A = 1.08 * E, this rearranges to `E > (1.96 / 0.08)^2 * 1.08 ≈ 700 expected claims`. Above roughly 700 expected claims, an A/E of 1.08 is statistically distinguishable from 1.0.

</details>

---

## Exercise 4: Gini drift z-test — discrimination vs calibration

**References:** Tutorial Part 7.

**What this exercise covers:** Falling Gini and rising A/E are different problems requiring different solutions. This exercise makes you compute Gini drift, interpret the DeLong z-test, and distinguish between the two failure modes.

### Task 1: Compute Gini drift

```python
gini_calc = GiniDrift()

y_ref = (df_reference["claim_count"] > 0).to_numpy().astype(int)
y_cur = (df_current["claim_count"] > 0).to_numpy().astype(int)

gini_result = gini_calc.calculate(
    y_ref=y_ref,
    pred_ref=pred_ref,
    y_cur=y_cur,
    pred_cur=pred_cur,
)

print(f"Gini (reference):  {gini_result.gini_ref:.4f}")
print(f"Gini (current):    {gini_result.gini_cur:.4f}")
print(f"Change:            {gini_result.gini_cur - gini_result.gini_ref:+.4f}")
print(f"Z-statistic:       {gini_result.z_stat:.4f}")
print(f"P-value:           {gini_result.p_value:.4f}")
print(f"Traffic light:     {gini_result.traffic_light}")
```

### Task 2: Plot the ROC curves

Plot reference and current ROC curves side by side. Annotate each with its Gini.

```python
from sklearn.metrics import roc_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y, pred, label, colour, gini in [
    (axes[0], y_ref, pred_ref, "Reference", "steelblue", gini_result.gini_ref),
    (axes[1], y_cur, pred_cur, "Current",   "tomato",    gini_result.gini_cur),
]:
    fpr, tpr, _ = roc_curve(y, pred)
    ax.plot(fpr, tpr, color=colour, linewidth=2,
            label=f"{label} (Gini={gini:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {label} period")
    ax.legend()

plt.suptitle(
    f"Gini drift: {gini_result.gini_ref:.3f} → {gini_result.gini_cur:.3f}  "
    f"(p = {gini_result.p_value:.3f})",
    fontsize=12,
)
plt.tight_layout()
plt.show()
```

### Task 3: Segment Gini analysis

Compute Gini separately for drivers aged 17–24 and 40–59. For each segment, print the reference Gini, current Gini, absolute change, and p-value.

```python
for low, high, label in [(17, 25, "17-24"), (40, 60, "40-59")]:
    ref_mask = (
        (df_reference["driver_age"] >= low) & (df_reference["driver_age"] < high)
    ).to_numpy()
    cur_mask = (
        (df_current["driver_age"] >= low) & (df_current["driver_age"] < high)
    ).to_numpy()

    seg_result = gini_calc.calculate(
        y_ref   = y_ref[ref_mask],
        pred_ref= pred_ref[ref_mask],
        y_cur   = y_cur[cur_mask],
        pred_cur= pred_cur[cur_mask],
    )
    print(
        f"{label}: ref={seg_result.gini_ref:.4f}, "
        f"cur={seg_result.gini_cur:.4f}, "
        f"change={seg_result.gini_cur - seg_result.gini_ref:+.4f}, "
        f"p={seg_result.p_value:.4f}"
    )
```

### Task 4: The two failure modes

**Question 4a:** The tutorial describes two distinct model failure modes: (1) calibration failure — A/E is off but Gini is stable; (2) discrimination failure — Gini is falling, A/E may or may not be off. Using your results from Exercises 3 and 4, classify the current state of the model. Which failure mode, if either, is present?

**Question 4b:** Suppose you observe A/E = 1.12 (CI excludes 1.0) and Gini has fallen by 0.04 (p = 0.02). Write out the appropriate response in order of steps, explaining what applying a recalibration factor in isolation would and would not fix.

**Question 4c:** The DeLong test compares AUCs rather than Ginis directly. Gini = 2 * AUC - 1. Does the direction of the conversion matter when interpreting the p-value? Explain in one sentence.

<details>
<summary>Solution — Exercise 4</summary>

**Answer to 4a:** Inspect the results from your run. If `ae_portfolio.traffic_light` is GREEN and `gini_result.p_value` > 0.05, the model is performing as expected — no failure mode active. If A/E is amber with CI excluding 1.0 but Gini is stable (p > 0.10), this is calibration failure only. If Gini p-value < 0.05 and the drop exceeds 0.03, that is discrimination failure regardless of the A/E.

**Answer to 4b:** Step 1: Apply a temporary recalibration factor (1/1.12 ≈ 0.893) to restore the portfolio average to correct calibration. Document the factor, the trigger, and the date in the recalibration history table. Step 2: Immediately initiate a retraining assessment. A falling Gini means the model's risk ranking is degrading — applying a multiplicative scalar to all predictions preserves the ranking, so the recalibration does nothing to fix the discrimination problem. Step 3: Schedule a model retraining on data from the current period. Step 4: Escalate to the head of pricing if the Gini drop persists into the following month.

**Answer to 4c:** No — since Gini = 2 * AUC - 1 is a monotone transformation, a statistically significant difference in AUCs is identical to a statistically significant difference in Ginis; the p-value is unchanged by the rescaling.

</details>

---

## Exercise 5: MonitoringReport — traffic light interpretation

**References:** Tutorial Parts 8 and 9.

**What this exercise covers:** The `MonitoringReport` aggregates all metrics into a traffic light system. This exercise makes you assemble a report, read the overall status correctly, and work through the five interpretation scenarios from Part 9.

### Task 1: Assemble and print the report

```python
report = MonitoringReport(
    model_name="motor_frequency_v1",
    reference_date="2022-12-31",
    current_date="2023-12-31",
)

report.add_psi(psi_result)

for feature, result in csi_results.items():
    report.add_csi(result)

report.add_ae(ae_portfolio)
report.add_gini_drift(gini_result)

summary = report.summary()

print("=" * 60)
print(f"OVERALL STATUS: {summary['overall_traffic_light']}")
print("=" * 60)
print()

m = summary["metrics"]
print(f"{'Score PSI':<35} {m['psi_score']['value']:>8.4f}  {m['psi_score']['traffic_light']}")
print(f"{'A/E ratio':<35} {m['ae_ratio']['value']:>8.4f}  {m['ae_ratio']['traffic_light']}")
print(f"{'A/E 95% CI':<35} [{m['ae_ratio']['ci_lower']:.4f}, {m['ae_ratio']['ci_upper']:.4f}]")
print(f"{'Gini (current)':<35} {m['gini']['gini_cur']:>8.4f}  {m['gini']['traffic_light']}")
print(f"{'Gini p-value':<35} {m['gini']['p_value']:>8.4f}")
print()
print("Feature CSI:")
for csi_item in sorted(summary["csi"], key=lambda x: x["csi"], reverse=True):
    print(f"  {csi_item['feature']:<28} {csi_item['csi']:>8.4f}  {csi_item['traffic_light']}")
```

### Task 2: The aggregation logic

**Question 2:** The overall traffic light is not simply the worst individual metric. Reproduce the aggregation rules from the tutorial and show that they produce the correct overall status for the following four hypothetical scenarios. Fill in the "Overall" column.

| Scenario | Score PSI | A/E | Gini | Max CSI | Overall |
|----------|-----------|-----|------|---------|---------|
| A | GREEN | GREEN | GREEN | GREEN | ? |
| B | AMBER | GREEN | GREEN | GREEN | ? |
| C | AMBER | AMBER | GREEN | GREEN | ? |
| D | GREEN | GREEN | RED | GREEN | ? |

Write the logic as a Python function and test it against all four scenarios:

```python
def overall_traffic_light(
    psi_tl: str,
    ae_tl: str,
    gini_tl: str,
    max_csi_tl: str,
) -> str:
    """
    Reproduce the MonitoringReport aggregation rules:
    - Any RED → overall RED
    - Two or more AMBER → overall AMBER
    - One AMBER (rest GREEN) → overall AMBER
    - All GREEN → overall GREEN
    """
    # Your implementation here
    pass


scenarios = [
    ("GREEN", "GREEN", "GREEN", "GREEN"),
    ("AMBER", "GREEN", "GREEN", "GREEN"),
    ("AMBER", "AMBER", "GREEN", "GREEN"),
    ("GREEN", "GREEN", "RED",   "GREEN"),
]

for psi, ae, gini, csi in scenarios:
    result = overall_traffic_light(psi, ae, gini, csi)
    print(f"PSI={psi}, A/E={ae}, Gini={gini}, CSI={csi}  →  {result}")
```

### Task 3: Scenario interpretation

For each of the five Part 9 scenarios, state in one sentence the appropriate action. Then match each scenario to the combination of metric states that produces it:

| Scenario | Description | Metric combination |
|----------|-------------|--------------------|
| 1 | All green | PSI=?, A/E=?, Gini=? |
| 2 | Elevated PSI, green A/E, stable Gini | PSI=?, A/E=?, Gini=? |
| 3 | Elevated PSI, elevated A/E, stable Gini | PSI=?, A/E=?, Gini=? |
| 4 | Green PSI, elevated A/E, falling Gini | PSI=?, A/E=?, Gini=? |
| 5 | Elevated CSI on one feature, all else green | PSI=?, A/E=?, Gini=? |

**Question 3:** Which scenario is the most serious, and why? Which scenario requires a recalibration factor as opposed to a retraining? Which scenario requires neither?

### Task 4: What your actual report says

**Question 4:** Using the report you assembled in Task 1, state which of the five scenarios your model is in. Justify your answer by citing the specific metric values, not just the traffic light colours. If the model is in Scenario 2 or 3, state what the recalibration factor would be and compute it.

<details>
<summary>Solution — Exercise 5</summary>

```python
# Task 2: Overall traffic light aggregation
def overall_traffic_light(
    psi_tl: str,
    ae_tl: str,
    gini_tl: str,
    max_csi_tl: str,
) -> str:
    metrics = [psi_tl, ae_tl, gini_tl, max_csi_tl]
    if "RED" in metrics:
        return "RED"
    amber_count = sum(1 for m in metrics if m == "AMBER")
    if amber_count >= 1:
        return "AMBER"
    return "GREEN"


# Note: the tutorial states "one AMBER → overall AMBER" (not "two or more").
# One AMBER with the rest GREEN is still AMBER. Verify:
for psi, ae, gini, csi in scenarios:
    result = overall_traffic_light(psi, ae, gini, csi)
    print(f"PSI={psi}, A/E={ae}, Gini={gini}, CSI={csi}  →  {result}")
# A → GREEN, B → AMBER, C → AMBER, D → RED
```

**Answer to Question 3:**
- Scenario 4 is the most serious: concept drift. The model's features have lost predictive power, which means the risk ranking is wrong. Recalibration cannot fix it.
- Scenarios 2 and 3 are calibration problems: Scenario 3 requires a recalibration factor; Scenario 2 requires no model action (the model is handling the mix shift correctly).
- Scenario 1 requires no action. Scenario 5 requires investigation of the feature but no immediate model action.

</details>

---

## Exercise 6: Recalibration decision-making

**References:** Tutorial Part 13.

**What this exercise covers:** Deciding whether to recalibrate, retrain, or wait is the hardest part of monitoring. This exercise gives you four months of fabricated monitoring history and asks you to apply the trigger logic mechanically, then justify whether the automated decision matches your judgement.

### Task 1: Implement the recalibration recommendation function

Write `recalibration_recommendation` based on the trigger thresholds from Tutorial Part 13. It must handle the case where there is no history (first monitoring run).

```python
def recalibration_recommendation(
    ae_ratio: float,
    ae_ci_lower: float,
    ae_ci_upper: float,
    gini_p_value: float,
    gini_change: float,
    history: pl.DataFrame | None,
) -> dict:
    """
    Apply the trigger logic from Tutorial Part 13.

    Parameters
    ----------
    ae_ratio     : float  — current month's A/E point estimate
    ae_ci_lower  : float  — lower bound of 95% CI
    ae_ci_upper  : float  — upper bound of 95% CI
    gini_p_value : float  — p-value from DeLong test
    gini_change  : float  — gini_cur - gini_ref (negative = declining)
    history      : pl.DataFrame or None
        Columns: current_date, ae_ci_lower, ae_ci_upper
        Rows ordered most-recent first.
        If None, treat as first run (no history).

    Returns
    -------
    dict with keys "recommendation" and "reason".
    Recommendation is one of: NO_ACTION, WATCH, RECALIBRATE, RETRAIN.
    """
    # Your implementation here.
    pass
```

Test it against the four scenarios below:

```python
# Fabricated monitoring history (four months, most-recent first)
history_4mo = pl.DataFrame({
    "current_date": ["2023-10-31", "2023-09-30", "2023-08-31", "2023-07-31"],
    "ae_ratio":     [1.07,         1.06,          1.05,          1.03],
    "ae_ci_lower":  [1.02,         1.01,          0.99,          0.97],
    "ae_ci_upper":  [1.12,         1.11,          1.11,          1.09],
})

test_cases = [
    # (ae_ratio, ci_lo, ci_hi, gini_p, gini_chg, history,       description)
    (1.08, 1.03, 1.13,  0.40,  -0.01,  None,           "First run, mild over-prediction"),
    (1.08, 1.03, 1.13,  0.40,  -0.01,  history_4mo,    "Two months CI excludes 1.0"),
    (1.18, 1.12, 1.24,  0.40,  -0.01,  None,           "A/E outside [0.90, 1.10]"),
    (1.06, 1.01, 1.11,  0.02,  -0.04,  None,           "Significant Gini drop"),
]

print(f"{'Description':<45} {'Recommendation':<15}  Reason")
print("-" * 95)
for ae, lo, hi, gp, gc, hist, desc in test_cases:
    rec = recalibration_recommendation(ae, lo, hi, gp, gc, hist)
    print(f"{desc:<45} {rec['recommendation']:<15}  {rec['reason']}")
```

**Expected output:**
```
First run, mild over-prediction          WATCH            A/E CI excludes 1.0 - monitor next month
Two months CI excludes 1.0               RECALIBRATE      A/E CI excludes 1.0 for 2+ months
A/E outside [0.90, 1.10]                RETRAIN          A/E outside [0.90, 1.10]
Significant Gini drop                    RETRAIN          Statistically significant Gini drop
```

### Task 2: Compute the recalibration factor from your actual data

```python
if ae_portfolio.ratio > 0:
    recal_factor = 1.0 / ae_portfolio.ratio
    print(f"A/E ratio:            {ae_portfolio.ratio:.4f}")
    print(f"Recalibration factor: {recal_factor:.4f}")
    print(f"Interpretation: multiply all frequency predictions by {recal_factor:.4f}")
    print()
    # Apply and verify
    expected_recal = expected_cur * recal_factor
    ae_after = expected_recal.sum() / actual_cur.sum()  # This should be ~1.0... wait
    # Note: recal_factor = 1/A/E, so new A/E = actual / (expected * (1/ae)) = ae / ae = 1
    ae_after_correct = actual_cur.sum() / expected_recal.sum()
    print(f"A/E after recalibration: {ae_after_correct:.4f}  (should be 1.0000)")
```

**Question 2:** The recalibration factor is 1/A/E. Explain why this is a **portfolio-level** fix only. Give a numerical example with two segments that illustrates how the overall A/E becomes 1.0 while both segment-level A/Es remain wrong.

### Task 3: The "recalibrated twice without resolution" trigger

**Question 3:** The tutorial states that if a recalibration factor has been applied twice without resolving the A/E signal, the recommendation escalates to RETRAIN. Why is this rule necessary? What does it tell you about the nature of the underlying drift if recalibration keeps failing to restore the A/E?

<details>
<summary>Solution — Exercise 6</summary>

```python
# Task 1: Recalibration recommendation
def recalibration_recommendation(
    ae_ratio: float,
    ae_ci_lower: float,
    ae_ci_upper: float,
    gini_p_value: float,
    gini_change: float,
    history: pl.DataFrame | None,
) -> dict:
    ae_ci_excludes_1  = (ae_ci_lower > 1.0) or (ae_ci_upper < 1.0)
    ae_outside_10pct  = (ae_ratio < 0.90) or (ae_ratio > 1.10)
    gini_significant  = (gini_p_value < 0.05) and (abs(gini_change) >= 0.03)

    # Count consecutive months where CI excluded 1.0
    consecutive_amber = 0
    if history is not None:
        for row in history.iter_rows(named=True):
            if row["ae_ci_lower"] > 1.0 or row["ae_ci_upper"] < 1.0:
                consecutive_amber += 1
            else:
                break  # Stop at first month where CI contained 1.0

    if ae_outside_10pct:
        return {"recommendation": "RETRAIN",
                "reason": "A/E outside [0.90, 1.10]"}
    elif gini_significant:
        return {"recommendation": "RETRAIN",
                "reason": "Statistically significant Gini drop"}
    elif ae_ci_excludes_1 and consecutive_amber >= 1:
        # current month + at least one prior = 2+ consecutive
        return {"recommendation": "RECALIBRATE",
                "reason": "A/E CI excludes 1.0 for 2+ months"}
    elif ae_ci_excludes_1:
        return {"recommendation": "WATCH",
                "reason": "A/E CI excludes 1.0 - monitor next month"}
    else:
        return {"recommendation": "NO_ACTION",
                "reason": "All signals within tolerance"}
```

**Answer to Question 2:** Consider two segments, A and B:
- Segment A: 400 actual, 350 expected → A/E = 1.143
- Segment B: 200 actual, 250 expected → A/E = 0.800
- Portfolio: 600 actual, 600 expected → A/E = 1.000

After recalibrating by 1/1.0 = 1.0 (the portfolio A/E happens to be 1.0 because errors cancel), nothing changes. Now consider portfolio A/E = 1.08: recal_factor = 0.926. Segment A new expected = 350 * 0.926 = 324, A/E = 400/324 = 1.235. Segment B new expected = 250 * 0.926 = 231, A/E = 200/231 = 0.866. Portfolio: 600 / 555 = 1.081... then rounds to 1.0 — but only at the level that the factor was computed. The segments are still wrong, just uniformly scaled.

**Answer to Question 3:** If recalibration is applied but the A/E drifts back above the trigger threshold within a month or two, the drift is not a fixed offset but a time-varying one — the relationship between features and outcomes is continuing to change. A multiplicative scalar addresses a static level shift; a dynamic drift means the model's feature weightings are increasingly wrong and no static correction will keep up. Repeated recalibration failures are therefore diagnostic of concept drift, not simple calibration error.

</details>

---

## Exercise 7: Building a regulatory evidence pack

**References:** Tutorial Part 15.

**What this exercise covers:** SS1/23 requires documented evidence that monitoring happened, that thresholds are defined, and that breaches triggered a response. This exercise makes you generate a monitoring framework document and an annual summary from fabricated multi-month data.

### Task 1: Write the monitoring framework document

Write a function `generate_framework_doc` that produces a formatted string describing the monitoring setup. It must include all five elements the tutorial identifies as required for a PRA review.

```python
def generate_framework_doc(
    model_name: str,
    model_version: str,
    reference_date: str,
    run_date: str,
) -> str:
    """
    Generate the monitoring framework documentation string.
    Must cover: (1) metrics and thresholds, (2) monitoring schedule,
    (3) action triggers, (4) data windows, (5) responsibility.
    """
    # Your implementation here.
    pass


doc = generate_framework_doc(
    model_name="motor_frequency_v1",
    model_version="1",
    reference_date="2022-12-31",
    run_date="2023-12-31",
)
print(doc)
assert "PSI" in doc
assert "Gini" in doc
assert "recalibrat" in doc.lower()
assert "escalat" in doc.lower()
```

### Task 2: Generate an annual summary from multi-month monitoring history

The cell below provides 12 months of fabricated monitoring results for 2023. Use them to generate the annual evidence pack summary.

```python
# 12 months of fabricated monitoring results
annual_history = pl.DataFrame({
    "current_date": [
        "2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30",
        "2023-05-31", "2023-06-30", "2023-07-31", "2023-08-31",
        "2023-09-30", "2023-10-31", "2023-11-30", "2023-12-31",
    ],
    "overall_traffic_light": [
        "GREEN", "GREEN", "GREEN", "GREEN",
        "GREEN", "AMBER", "AMBER", "AMBER",
        "AMBER", "AMBER", "GREEN", "GREEN",
    ],
    "ae_ratio": [
        1.01, 0.99, 1.02, 1.00,
        1.03, 1.06, 1.07, 1.09,
        1.08, 1.07, 1.03, 1.01,
    ],
    "ae_ci_lower": [
        0.96, 0.94, 0.97, 0.95,
        0.98, 1.01, 1.02, 1.04,
        1.03, 1.02, 0.98, 0.96,
    ],
    "ae_ci_upper": [
        1.06, 1.04, 1.07, 1.05,
        1.08, 1.11, 1.12, 1.14,
        1.13, 1.12, 1.08, 1.06,
    ],
    "gini_cur": [
        0.421, 0.418, 0.420, 0.419,
        0.416, 0.415, 0.413, 0.410,
        0.412, 0.414, 0.418, 0.420,
    ],
    "gini_p_value": [
        0.72, 0.68, 0.74, 0.70,
        0.55, 0.31, 0.28, 0.19,
        0.22, 0.25, 0.48, 0.65,
    ],
    "psi_score": [
        0.04, 0.05, 0.04, 0.06,
        0.07, 0.09, 0.11, 0.12,
        0.10, 0.09, 0.07, 0.05,
    ],
    "action_taken": [
        None, None, None, None,
        None, None, None, "RECALIBRATE",
        None, None, None, None,
    ],
})

# Print the annual summary table
print(f"ANNUAL MONITORING SUMMARY - 2023")
print(f"Model: motor_frequency_v1")
print(f"Total runs: {annual_history.shape[0]}")
print()
print(f"{'Month':<12} {'Overall':<10} {'A/E':>8}  {'Gini':>8}  {'PSI':>8}  {'Action'}")
print("-" * 65)
for row in annual_history.iter_rows(named=True):
    action = row["action_taken"] or "-"
    print(
        f"{row['current_date']:<12} {row['overall_traffic_light']:<10} "
        f"{row['ae_ratio']:>8.4f}  {row['gini_cur']:>8.4f}  "
        f"{row['psi_score']:>8.4f}  {action}"
    )

# Traffic light distribution
tl_counts = annual_history.group_by("overall_traffic_light").len().sort("overall_traffic_light")
print()
print("Traffic light distribution:")
for row in tl_counts.iter_rows(named=True):
    print(f"  {row['overall_traffic_light']}: {row['len']} months")
```

### Task 3: Breach response narrative

**Question 3a:** Looking at the annual history table, identify the breach period precisely: which months triggered AMBER status? According to the tutorial's trigger rules, in which month should the recalibration have been triggered, and was it applied at the right time? The `action_taken` column records that recalibration was applied in August 2023. Was that correct, early, or late per the rules?

**Question 3b:** A PRA reviewer asks: "You had five consecutive AMBER months. What action did you take, and when?" Write a one-paragraph breach response in the style of a regulatory submission. It must cite the specific trigger condition, the date the recalibration factor was applied, the factor value (use ae_ratio from the month prior to the action), and the outcome (the signal resolved in November 2023).

**Question 3c:** The tutorial notes that the breach response log and recalibration history must be stored in governed Delta tables, not in `/dbfs/tmp/`. Write the SQL queries (as strings in Python — no Spark required) that a PRA reviewer would run to verify: (1) the monitoring ran on schedule in 2023, (2) any recalibrations were applied, and (3) the monitoring framework version in effect during the breach period.

```python
queries = {
    "monitoring_schedule": """
        -- Query 1: Verify monitoring ran monthly in 2023
        -- Your SQL here
    """,
    "recalibrations": """
        -- Query 2: List all recalibrations applied to this model
        -- Your SQL here
    """,
    "framework_version": """
        -- Query 3: Framework version in effect during August 2023
        -- Your SQL here
    """,
}

for name, sql in queries.items():
    print(f"--- {name} ---")
    print(sql.strip())
    print()
```

<details>
<summary>Solution — Exercise 7</summary>

```python
# Task 1: Framework document
def generate_framework_doc(
    model_name: str,
    model_version: str,
    reference_date: str,
    run_date: str,
) -> str:
    return f"""
MOTOR FREQUENCY MODEL — MONITORING FRAMEWORK
=============================================
Model:             {model_name}
Version:           {model_version}
Framework version: 1.0
Effective from:    {reference_date}
Last reviewed:     {run_date}

MONITORING SCHEDULE
-------------------
Frequency:    Monthly (automated job)
Run timing:   1st of month at 06:00 UK time
Notebook:     module-11-model-monitoring

METRICS AND THRESHOLDS
----------------------
1. Score PSI (Population Stability Index)
   Green:  PSI < 0.10
   Amber:  0.10 <= PSI < 0.20
   Red:    PSI >= 0.20

2. A/E Ratio (Actual vs Expected claim frequency)
   Green:  95% CI contains 1.0
   Amber:  95% CI excludes 1.0, ratio in [0.90, 1.10]
   Red:    Ratio outside [0.90, 1.10]

3. Gini Drift (DeLong z-test on AUC)
   Green:  p-value > 0.10, or Gini drop < 0.03
   Amber:  p-value 0.05-0.10, or p < 0.05 and drop < 0.03
   Red:    p-value < 0.05 and Gini drop >= 0.03

4. Feature CSI (per feature)
   Green:  CSI < 0.10
   Amber:  0.10 <= CSI < 0.20
   Red:    CSI >= 0.20

OVERALL STATUS
--------------
Red:    Any single metric is Red
Amber:  Any metric is Amber (no Red)
Green:  All metrics Green

ACTION TRIGGERS
---------------
Recalibrate: A/E CI excludes 1.0 for two consecutive months, OR
             A/E point estimate outside [0.90, 1.10]
Retrain:     Significant Gini drop (p < 0.05, drop > 0.03) for two
             consecutive months, OR A/E outside [0.85, 1.15], OR
             Recalibration applied twice without resolving A/E signal
Escalate:    Overall Red in any month — immediate escalation to CRO
             Three consecutive Amber months — review within 5 working days

DATA WINDOWS
------------
Reference: Rolling 12 months prior to current period
Current:   Previous calendar month
Minimum policies for valid report: 1,000
Minimum claims for Gini reliability: 50

RESPONSIBILITY
--------------
Owner:          Head of Pricing
Monthly review: Named pricing analyst (see job configuration)
Escalation:     CRO / Actuarial Function Holder
""".strip()
```

**Answer to Question 3a:** AMBER status began in June 2023 and ran through October 2023 (five consecutive months). Per the trigger rules, recalibration should be applied when the A/E CI has excluded 1.0 for **two consecutive months**. The CI first excluded 1.0 in June 2023 (ci_lower = 1.01). The second consecutive month where CI excluded 1.0 was July 2023 (ci_lower = 1.02). Therefore the recalibration trigger was hit at the July 2023 monitoring run. Recalibration was applied in August 2023 — one month late by the strict rule (it should have been applied at the July run or immediately after it). In practice a one-month lag is common if monitoring runs at the start of the month and recalibration goes through a governance approval step, but the documentation should acknowledge the timing.

**Answer to Question 3b (sample regulatory submission language):**

> The motor frequency model (motor_frequency_v1, version 1) entered AMBER status in June 2023, when the A/E ratio was 1.06 with a 95% confidence interval of [1.01, 1.11], which no longer contained 1.0. The amber status persisted into July 2023 (A/E 1.07, CI [1.02, 1.12]), triggering the two-consecutive-month recalibration criterion defined in the monitoring framework. A recalibration factor of 0.935 (= 1 / 1.07, based on the July 2023 A/E) was applied to the model output effective 1 August 2023. This was recorded in the recalibration_history Delta table. The A/E signal resolved by November 2023 (A/E 1.03, CI [0.98, 1.08], containing 1.0), at which point the model returned to GREEN status. No further action was required. A retraining assessment was initiated in September 2023 as a precaution; the assessment concluded that the drift was calibration-related rather than discriminatory (Gini remained stable throughout), and retraining was not recommended.

```python
# Answer to Question 3c
queries = {
    "monitoring_schedule": """
        SELECT current_date, overall_traffic_light, ae_ratio, psi_score
        FROM main.motor_monitoring.monitoring_log
        WHERE model_name = 'motor_frequency_v1'
          AND YEAR(current_date) = 2023
        ORDER BY current_date;
    """,
    "recalibrations": """
        SELECT effective_from, recalibration_factor, ae_ratio_at_trigger,
               reason, applied_by
        FROM main.motor_monitoring.recalibration_history
        WHERE model_name = 'motor_frequency_v1'
        ORDER BY effective_from;
    """,
    "framework_version": """
        SELECT version, effective_from, content
        FROM main.motor_monitoring.monitoring_framework_versions
        WHERE model_name = 'motor_frequency_v1'
          AND effective_from <= '2023-08-31'
        ORDER BY effective_from DESC
        LIMIT 1;
    """,
}
```

</details>
