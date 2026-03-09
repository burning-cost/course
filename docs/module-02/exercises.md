# Module 2 Exercises: GLMs in Python - The Bridge from Emblem

Four exercises. Work through them in order - each builds on the previous.

Before starting, make sure you have run all the cells in `tutorial.md` from top to bottom in your notebook. The exercises assume the `df`, `df_pd`, `glm_freq`, `freq_rels`, `glm_sev`, and `sev_rels` variables are defined in your session.

If your cluster restarted since running the tutorial, you will need to re-run the cells from the top. Variables do not persist across cluster restarts.

---

## Exercise 1: Base level mismatch - the classic validation failure

**What you are practising:** The most common error in GLM migrations. Understanding it once means you will never make it in production.

**The scenario:** Your colleague has sent you the Emblem CSV export for a motor frequency model. Your job is to validate that your Python model agrees with it.

```csv
Factor,Level,Relativity,SE
area,B,1.1041,0.0131
area,C,1.2185,0.0122
area,D,1.4173,0.0143
area,E,1.6471,0.0172
area,F,1.9120,0.0201
ncd_years,0,1.0000,0.0000
ncd_years,1,0.8826,0.0141
ncd_years,2,0.8089,0.0126
ncd_years,3,0.7239,0.0119
ncd_years,4,0.6618,0.0108
ncd_years,5,0.5232,0.0092
conviction_flag,0,1.0000,0.0000
conviction_flag,1,1.5220,0.0418
```

Area A is not in the CSV because it is the implicit base level - Emblem does not export rows for base levels, just as Emblem's factor charts show area A as a horizontal line at 1.0.

**Before you start:** Copy the CSV content above into a file called `emblem_export.csv` in your notebook's working directory. The simplest way is to create it directly in a notebook cell:

```python
emblem_csv = """Factor,Level,Relativity,SE
area,B,1.1041,0.0131
area,C,1.2185,0.0122
area,D,1.4173,0.0143
area,E,1.6471,0.0172
area,F,1.9120,0.0201
ncd_years,0,1.0000,0.0000
ncd_years,1,0.8826,0.0141
ncd_years,2,0.8089,0.0126
ncd_years,3,0.7239,0.0119
ncd_years,4,0.6618,0.0108
ncd_years,5,0.5232,0.0092
conviction_flag,0,1.0000,0.0000
conviction_flag,1,1.5220,0.0418"""

with open("/tmp/emblem_export.csv", "w") as f:
    f.write(emblem_csv)

print("Emblem CSV saved to /tmp/emblem_export.csv")
```

**Task 1.** Fit the Poisson GLM with `ncd_years` and `conviction_flag` as **continuous** covariates - no `C()` wrapper around them. Use this formula:

```python
wrong_formula = "claim_count ~ C(area) + ncd_years + conviction_flag"
```

Run the model. Then compute what the NCD=1, NCD=2, ..., NCD=5 relativities look like from this formula - for a continuous covariate, the relativity at value k is `exp(coefficient × k)`. Compare these to the Emblem CSV above.

**Task 2.** Explain in your own words why the comparison fails for NCD and conviction. The hint: look at the number of parameters. How many numbers does the continuous model use to describe NCD's effect? How many does the categorical model use?

**Task 3.** Refit with the correct categorical encoding:

```python
correct_formula = (
    "claim_count ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)
```

Extract relativities using `extract_freq_relativities()` from the tutorial. Compare area, NCD, and conviction relativities to the Emblem CSV. Are they within 1% for all levels?

**Task 4.** Use the `compare_to_emblem()` function from Part 8 of the tutorial. The function takes a Polars DataFrame of Python relativities and a path to the Emblem CSV. Run it on your corrected model. What is the maximum relative difference? What is the most likely explanation for any remaining gap?

**What to look for:** If you see a uniform multiplier across all levels of a factor (e.g. all NCD relativities are 3% higher in Python than Emblem), the base level is wrong on one side. Divide any Python relativity by the corresponding Emblem relativity: if all quotients for a given factor are equal to the same constant, the base level differs.

---

### Solution - Exercise 1

```python
import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

# Task 1: Wrong formula - ncd_years and conviction_flag as continuous
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_wrong = smf.glm(
        formula="claim_count ~ C(area) + ncd_years + conviction_flag",
        data=df_pd,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd["log_exposure"],
    ).fit()

# Compute relativities at each NCD level from the continuous slope
ncd_slope = glm_wrong.params["ncd_years"]
emblem_ncd = {0: 1.0000, 1: 0.8826, 2: 0.8089, 3: 0.7239, 4: 0.6618, 5: 0.5232}

print("Task 1: NCD relativities from continuous model vs Emblem categorical:")
print(f"{'NCD':<5} {'Continuous_GLM':>16} {'Emblem_Categorical':>20} {'Match?':>8}")
for k in range(6):
    cont_rel = np.exp(ncd_slope * k)
    emb_rel = emblem_ncd[k]
    match = abs(cont_rel / emb_rel - 1) < 0.01
    print(f"{k:<5} {cont_rel:>16.4f} {emb_rel:>20.4f} {str(match):>8}")

print(f"\nContinuous NCD slope: {ncd_slope:.4f}")
print("The continuous model fits a single best-fit slope, not free per-level estimates.")
```

**Task 2 answer:** The continuous model uses one number to describe NCD's effect (the slope). The categorical model uses five numbers (one coefficient per non-base level). These will agree only if the relationship between NCD and log-frequency is perfectly linear, which the real data does not guarantee. In this synthetic data, the DGP is exactly linear in NCD (-0.13 per year), so the continuous model is approximately correct. In real data, the NCD effect is rarely perfectly linear across all levels.

```python
# Task 3: Correct formula - categorical NCD and conviction
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_correct = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group"
        ),
        data=df_pd,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd["log_exposure"],
    ).fit()

# Extract relativities
correct_rels = extract_freq_relativities(
    glm_correct,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)

print("Task 3: NCD relativities - corrected Python model vs Emblem:")
print(f"{'NCD':<5} {'Python_GLM':>12} {'Emblem':>10} {'% diff':>10}")
for k in range(1, 6):
    py_rel = correct_rels.filter(
        (pl.col("feature") == "ncd_years") & (pl.col("level") == str(k))
    )["relativity"].item()
    emb_rel = emblem_ncd[k]
    pct = (py_rel / emb_rel - 1) * 100
    print(f"{k:<5} {py_rel:>12.4f} {emb_rel:>10.4f} {pct:>9.2f}%")

conv_rel = correct_rels.filter(
    (pl.col("feature") == "conviction_flag") & (pl.col("level") == "1")
)["relativity"].item()
print(f"\nConviction: Python={conv_rel:.4f}  Emblem=1.5220  Diff={(conv_rel/1.5220-1)*100:.2f}%")
```

```python
# Task 4: compare_to_emblem function (from Part 8 of tutorial)
# Copy the function definition here if not already in scope, then run:
emblem_comparison = compare_to_emblem(correct_rels, "/tmp/emblem_export.csv", tolerance=0.01)
```

**Task 4 discussion.** After correcting the encoding, relativities should match to within 1% for all factor levels. Any residual gap above 1% almost certainly means one of:

- The two models were not fit on identical data (different data extract dates, different filtering)
- Emblem has applied a credibility adjustment to sparse cells
- A manual override in Emblem - check whether any Emblem relativity is a suspiciously round number

If you see a uniform multiplier across all levels of a factor, the base level is still wrong on one side. Divide any Python relativity by the corresponding Emblem relativity: if all quotients for a given factor are equal to the same constant, the base level differs.

---

## Exercise 2: Handling missing values

**What you are practising:** Understanding the difference between Emblem's missing value treatment and statsmodels' default - and how to handle missing values explicitly so both tools work on the same data.

**The scenario:** In real bordereaux data, vehicle group is often missing for mid-term policies, direct-to-consumer business with incomplete data feeds, or policies where the vehicle was changed mid-term and not properly recorded. You need to decide what to do before fitting.

**Setup:** Introduce 3% missing values in vehicle_group to simulate a data quality issue.

```python
rng2 = np.random.default_rng(seed=99)
missing_idx = rng2.choice(len(df), size=int(0.03 * len(df)), replace=False)

# Introduce missingness in Polars
df_missing = df.with_columns(
    pl.when(
        pl.int_range(pl.len()).is_in(missing_idx.tolist())
    )
    .then(None)
    .otherwise(pl.col("vehicle_group"))
    .alias("vehicle_group")
)

# Check: how many are now null?
print(f"Missing vehicle_group: {df_missing['vehicle_group'].is_null().sum():,}")

df_pd_missing = df_missing.to_pandas()
df_pd_missing["log_exposure"] = np.log(df_pd_missing["exposure"].clip(lower=1e-6))
```

**A new Python concept: `.is_in(list)`.** `pl.int_range(pl.len()).is_in(missing_idx.tolist())` generates a range of integers from 0 to n-1 (representing row indices) and checks whether each is in the list of missing indices. This is how you select specific rows by index in Polars.

**Task 1.** Fit the Poisson GLM on `df_pd_missing` without any missing value handling. Then check `glm_missing.nobs` (the number of observations statsmodels actually used) and compare it to `len(df_pd)`. How many rows were silently dropped?

**Task 2.** Impute the missing vehicle_group values with the column mean. Refit the model. Compare the conviction_flag relativity between:
- The full-data model (`glm_freq` from the tutorial)
- The model with missing values silently dropped
- The model with mean imputation

Are the differences material (more than 1%)? Why or why not?

**Task 3.** Create a `vg_missing` flag (1 = vehicle group was missing, 0 = known) and add it to the model as a predictor alongside the mean-imputed vehicle_group. Does the missingness indicator have a statistically significant coefficient? What would a significant coefficient tell you about the data?

**Task 4.** For a categorical factor, Emblem creates an "Unknown" level with its own estimated relativity. Introduce 5% missing values in `area` and create an "Unknown" level for them in Polars. Fit the model with "Unknown" as an area level. What is the estimated "Unknown" relativity and its confidence interval? What are the regulatory implications of pricing "Unknown" customers?

---

### Solution - Exercise 2

```python
import warnings

# Task 1: Silent row dropping
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_missing = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group"
        ),
        data=df_pd_missing,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd_missing["log_exposure"],
    ).fit()

print(f"Task 1:")
print(f"  Full dataset:     {len(df_pd):,} rows")
print(f"  Model actually used: {int(glm_missing.nobs):,} rows")
print(f"  Silently dropped: {len(df_pd) - int(glm_missing.nobs):,} rows ({(len(df_pd) - int(glm_missing.nobs)) / len(df_pd) * 100:.1f}%)")
print("  statsmodels dropped 3,000 rows without any warning or error message.")
```

```python
# Task 2: Mean imputation
vg_mean = df_missing["vehicle_group"].drop_nulls().mean()

df_imputed = df_missing.with_columns(
    pl.col("vehicle_group").fill_null(vg_mean).cast(pl.Int32).alias("vehicle_group")
)
df_pd_imputed = df_imputed.to_pandas()
df_pd_imputed["log_exposure"] = np.log(df_pd_imputed["exposure"].clip(lower=1e-6))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_imputed = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group"
        ),
        data=df_pd_imputed,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd_imputed["log_exposure"],
    ).fit()

conv_full     = np.exp(glm_freq.params["C(conviction_flag, Treatment(0))[T.1]"])
conv_dropped  = np.exp(glm_missing.params["C(conviction_flag, Treatment(0))[T.1]"])
conv_imputed  = np.exp(glm_imputed.params["C(conviction_flag, Treatment(0))[T.1]"])

print(f"Task 2: Conviction relativity across models")
print(f"  Full data (no missingness):    {conv_full:.4f}")
print(f"  Missing rows dropped silently: {conv_dropped:.4f}  (diff: {abs(conv_dropped/conv_full-1)*100:.2f}%)")
print(f"  Mean imputation:               {conv_imputed:.4f}  (diff: {abs(conv_imputed/conv_full-1)*100:.2f}%)")
print("  Mean imputation is approximately correct here because missingness is random.")
print("  In production, missingness is almost never truly random.")
```

```python
# Task 3: Missingness indicator
df_indicator = df_missing.with_columns([
    pl.col("vehicle_group").is_null().cast(pl.Int32).alias("vg_missing"),
    pl.col("vehicle_group").fill_null(vg_mean).cast(pl.Int32).alias("vehicle_group"),
])
df_pd_indicator = df_indicator.to_pandas()
df_pd_indicator["log_exposure"] = np.log(df_pd_indicator["exposure"].clip(lower=1e-6))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_indicator = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group + "
            "vg_missing"
        ),
        data=df_pd_indicator,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd_indicator["log_exposure"],
    ).fit()

vg_miss_coef = glm_indicator.params["vg_missing"]
vg_miss_pval = glm_indicator.pvalues["vg_missing"]

print(f"Task 3: Missingness indicator")
print(f"  Coefficient: {vg_miss_coef:.4f}  (relativity: {np.exp(vg_miss_coef):.4f})")
print(f"  p-value: {vg_miss_pval:.4f}")

if vg_miss_pval < 0.05:
    print("  SIGNIFICANT: missingness predicts frequency beyond vehicle group itself.")
    print("  This suggests the missing data is not missing at random - investigate why.")
else:
    print("  Not significant: missingness is approximately random.")
    print("  In production, always check - it is rarely truly random.")
```

**Task 3 interpretation:** If the missingness indicator has a significant coefficient, policies with missing vehicle group have systematically higher or lower claim frequency than their vehicle group alone would predict. This is a red flag: it suggests the data is missing for a reason that correlates with risk (e.g. vehicles with missing group data are more likely to be high-performance cars that the system could not classify). Pricing these policies as if missingness is random would be wrong.

```python
# Task 4: Unknown area level
rng3 = np.random.default_rng(seed=77)
area_missing_idx = rng3.choice(len(df), size=int(0.05 * len(df)), replace=False)

df_unknown = df.with_columns(
    pl.when(
        pl.int_range(pl.len()).is_in(area_missing_idx.tolist())
    )
    .then(pl.lit("Unknown"))
    .otherwise(pl.col("area").cast(pl.Utf8))
    .alias("area")
)

# Must include Unknown in the Enum ordering
df_unknown = df_unknown.with_columns(
    pl.col("area").cast(pl.Enum(["A", "B", "C", "D", "E", "F", "Unknown"]))
)

df_pd_unknown = df_unknown.to_pandas()
df_pd_unknown["log_exposure"] = np.log(df_pd_unknown["exposure"].clip(lower=1e-6))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_unknown = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area, Treatment('A')) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group"
        ),
        data=df_pd_unknown,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_pd_unknown["log_exposure"],
    ).fit()

unknown_key = "C(area, Treatment('A'))[T.Unknown]"
if unknown_key in glm_unknown.params:
    unknown_coef = glm_unknown.params[unknown_key]
    unknown_rel = np.exp(unknown_coef)
    unknown_ci_lo = np.exp(glm_unknown.conf_int().loc[unknown_key, 0])
    unknown_ci_hi = np.exp(glm_unknown.conf_int().loc[unknown_key, 1])
    print(f"\nTask 4: 'Unknown' area relativity: {unknown_rel:.4f}")
    print(f"  95% CI: [{unknown_ci_lo:.3f}, {unknown_ci_hi:.3f}]")
    print()
    print("  Regulatory note:")
    print("  Pricing 'Unknown' customers using a statistically estimated relativity")
    print("  is acceptable only if you can explain what drove the missingness.")
    print("  If missingness correlates with fraud or high-risk behaviour, the 'Unknown'")
    print("  relativity may under-price a systematically risky group.")
    print("  The FCA expects you to manage data quality, not price around it.")
```

---

## Exercise 3: Tweedie vs frequency-severity split

**What you are practising:** Understanding when these two approaches give different answers, and how to defend your choice to a technical reviewer or regulator.

**The scenario:** A junior analyst suggests using a Tweedie model "because it is one model instead of two and therefore simpler to audit." You need to test whether this simplification materially changes the area relativities, and prepare arguments for or against.

**The data setup we are working with:** The synthetic data was generated with area affecting **frequency only** (not severity) and vehicle group affecting **both** frequency and severity. This means the true pure premium area effect equals the true frequency area effect.

**Task 1.** Fit a Tweedie pure premium GLM using the same formula as the frequency GLM. Use `var_power=1.5` as the power parameter and `offset=log_exposure`. Extract the area relativities.

**Task 2.** Compare the Tweedie area relativities to the frequency GLM area relativities from the tutorial. Are they materially different? (They should not be, because area only affects frequency in this DGP.) What is the maximum percentage difference across all area levels?

**Task 3.** Now modify the severity data-generating process to add an area F severity effect of `exp(0.15) ≈ 1.16`:

```python
# Re-generate severity with area F having a severity uplift
rng_v2 = np.random.default_rng(seed=42)
# ... (regenerate all data, add 0.15 to log-severity for area F policies)
```

Re-fit the frequency-severity split and the Tweedie on this modified data. Compare the area F pure premium relativities from both models to the true value (`exp(0.65 + 0.15) = exp(0.80) ≈ 2.226`). Which model is closer?

**Task 4.** Write three counter-arguments to the analyst's claim that "Tweedie is simpler and easier to audit." Each argument should be specific - not "it's more complex" but a concrete reason why the split gives you something the Tweedie does not.

---

### Solution - Exercise 3

```python
import warnings

# Task 1: Tweedie pure premium model
df_pp = df.with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
)
df_pp_pd = df_pp.to_pandas()
df_pp_pd["log_exposure"] = np.log(df_pp_pd["exposure"].clip(lower=1e-6))

pp_formula = (
    "pure_premium ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_tweedie = smf.glm(
        formula=pp_formula,
        data=df_pp_pd,
        family=sm.families.Tweedie(
            var_power=1.5,
            link=sm.families.links.Log(),
        ),
        offset=df_pp_pd["log_exposure"],
    ).fit()

print(f"Tweedie GLM - Converged: {glm_tweedie.converged}, Iterations: {glm_tweedie.nit}")

tw_rels = extract_freq_relativities(
    glm_tweedie,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)

print("Tweedie area relativities:")
print(tw_rels.filter(pl.col("feature") == "area"))
```

```python
# Task 2: Compare to frequency GLM area relativities
true_area = {"A": 1.0, "B": np.exp(0.10), "C": np.exp(0.20),
             "D": np.exp(0.35), "E": np.exp(0.50), "F": np.exp(0.65)}

print("Task 2: Area relativities - Tweedie vs Frequency GLM vs True")
print(f"{'Area':<6} {'Tweedie':>10} {'Freq_GLM':>10} {'True':>10} {'Tweedie vs Freq':>16}")

max_diff = 0.0
for lv in ["A", "B", "C", "D", "E", "F"]:
    tw_rel = tw_rels.filter(
        (pl.col("feature") == "area") & (pl.col("level") == lv)
    )["relativity"].item()
    fr_rel = freq_rels.filter(
        (pl.col("feature") == "area") & (pl.col("level") == lv)
    )["relativity"].item()
    diff = abs(tw_rel / fr_rel - 1) * 100
    max_diff = max(max_diff, diff)
    print(f"{lv:<6} {tw_rel:>10.4f} {fr_rel:>10.4f} {true_area[lv]:>10.4f} {diff:>14.2f}%")

print(f"\nMaximum difference between Tweedie and Freq GLM: {max_diff:.2f}%")
print("With area only in the frequency DGP, both models recover approximately the same effect.")
```

```python
# Task 3: Add area to severity DGP
# Re-generate - we need a fresh rng to get the same base data, then add the area F sev effect
rng_v2 = np.random.default_rng(seed=42)
n = 100_000
area_v2 = rng_v2.choice(["A","B","C","D","E","F"], size=n, p=[0.10,0.18,0.25,0.22,0.15,0.10])
vehicle_group_v2 = rng_v2.integers(1, 51, size=n)
ncd_years_v2 = rng_v2.choice([0,1,2,3,4,5], size=n, p=[0.08,0.07,0.09,0.12,0.20,0.44])
driver_age_v2 = rng_v2.integers(17, 86, size=n)
conviction_flag_v2 = rng_v2.binomial(1, 0.06, size=n)
exposure_v2 = np.clip(rng_v2.beta(8, 2, size=n), 0.05, 1.0)

log_mu_freq_v2 = (
    -3.10
    + np.where(area_v2=="B", 0.10, 0) + np.where(area_v2=="C", 0.20, 0)
    + np.where(area_v2=="D", 0.35, 0) + np.where(area_v2=="E", 0.50, 0)
    + np.where(area_v2=="F", 0.65, 0)
    + 0.018 * (vehicle_group_v2 - 1) + (-0.13) * ncd_years_v2
    + np.where(driver_age_v2 < 25, 0.55, 0) + np.where(driver_age_v2 > 70, 0.28, 0)
    + 0.42 * conviction_flag_v2 + np.log(exposure_v2)
)
claim_count_v2 = rng_v2.poisson(np.exp(log_mu_freq_v2 - np.log(exposure_v2)) * exposure_v2)

# Severity DGP with area F uplift
sev_log_mu_v2 = np.log(3500) + 0.012 * (vehicle_group_v2 - 1) + np.where(area_v2=="F", 0.15, 0)
has_claim_v2 = claim_count_v2 > 0
true_mean_sev_v2 = np.exp(sev_log_mu_v2)
avg_sev_v2 = np.where(has_claim_v2, rng_v2.gamma(4.0, true_mean_sev_v2 / 4.0), 0.0)
incurred_v2 = avg_sev_v2 * claim_count_v2

df_v2 = pl.DataFrame({
    "area": area_v2,
    "vehicle_group": vehicle_group_v2,
    "ncd_years": ncd_years_v2,
    "driver_age": driver_age_v2,
    "conviction_flag": conviction_flag_v2,
    "exposure": exposure_v2,
    "claim_count": claim_count_v2,
    "avg_severity": avg_sev_v2,
    "incurred": incurred_v2,
}).with_columns(
    pl.col("area").cast(pl.Enum(["A","B","C","D","E","F"]))
)

df_v2_pd = df_v2.to_pandas()
df_v2_pd["log_exposure"] = np.log(df_v2_pd["exposure"].clip(lower=1e-6))
df_v2_pd["pure_premium"] = df_v2_pd["incurred"] / df_v2_pd["exposure"]

# True area F pure premium relativity = freq × sev = exp(0.65) × exp(0.15) = exp(0.80)
true_pp_F = np.exp(0.65 + 0.15)
print(f"True area F pure premium relativity: {true_pp_F:.4f}")

# Frequency GLM (same spec as before)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_freq_v2 = smf.glm(
        formula="claim_count ~ C(area) + C(ncd_years, Treatment(0)) + C(conviction_flag, Treatment(0)) + vehicle_group",
        data=df_v2_pd,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_v2_pd["log_exposure"],
    ).fit()

# Severity GLM (claimed only)
df_sev_v2 = df_v2.filter(pl.col("claim_count") > 0).to_pandas()
df_sev_v2["avg_severity"] = df_sev_v2["incurred"] / df_sev_v2["claim_count"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_sev_v2 = smf.glm(
        formula="avg_severity ~ C(area, Treatment('A')) + vehicle_group",
        data=df_sev_v2,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=df_sev_v2["claim_count"],
    ).fit()

freq_area_F = np.exp(glm_freq_v2.params["C(area)[T.F]"])
sev_area_F = np.exp(glm_sev_v2.params["C(area, Treatment('A'))[T.F]"])
split_pp_F = freq_area_F * sev_area_F

# Tweedie on v2 data
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_tw_v2 = smf.glm(
        formula=pp_formula,
        data=df_v2_pd,
        family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
        offset=df_v2_pd["log_exposure"],
    ).fit()

tw_area_F = np.exp(glm_tw_v2.params["C(area)[T.F]"])

print(f"\nArea F pure premium relativities:")
print(f"  True value:           {true_pp_F:.4f}")
print(f"  Frequency-sev split:  {split_pp_F:.4f}  (freq={freq_area_F:.4f} × sev={sev_area_F:.4f})")
print(f"  Tweedie:              {tw_area_F:.4f}")
print(f"\nBoth should be close to the true value of {true_pp_F:.4f}.")
print("The freq-sev split wins when you need SEPARATE freq and sev relativities - which you")
print("often do for reinsurance structuring and for explaining the model to stakeholders.")
```

**Task 4: Three counter-arguments to "Tweedie is simpler."**

First: "simpler" conflates model complexity with interpretability. The Tweedie gives you one set of pure premium relativities. It cannot tell you whether area F is expensive because drivers there have more accidents or because accidents there cost more to settle. That distinction matters for reinsurance structuring: frequency excess of loss treaties are designed around frequency patterns, large loss per-risk treaties are designed around severity patterns. A single Tweedie relativity strips out exactly the information needed for that decision.

Second: the Tweedie power parameter `p` is an additional assumption that the frequency-severity split does not require. When the regulator asks "why does your model treat compound frequency and severity jointly?" your answer needs to justify that power assumption. The frequency-severity split has a cleaner actuarial justification: claims arrive as a Poisson process and each claim has a Gamma-distributed cost. That story is easy to tell to a non-technical committee or FCA reviewer.

Third: the frequency-severity split allows different sets of rating factors for frequency and severity. Occupation might be a strong frequency predictor but irrelevant for severity. Including it in a Tweedie forces a joint effect that may be statistically wrong. With a split model, you can test factor significance independently in each component and only include factors where there is genuine evidence.

---

## Exercise 4: Model change log

**What you are practising:** Building the change management evidence that PS 21/5 and Consumer Duty require. Any material change to pricing relativities needs to be documented, explained, and signed off before production deployment.

**The scenario:** You have just fit a new motor frequency model (v2, which adds driver age flags) and need to compare it to the previous cycle's model (v1, without driver age). Your pricing committee requires sign-off on any factor relativity change above 5%.

**Setup:** Fit the v2 model with driver age flags included.

```python
import warnings

# v1: no driver age (already fitted as glm_freq in the tutorial)
# v2: add young_driver and old_driver flags
df_with_age = df.with_columns([
    (pl.col("driver_age") < 25).cast(pl.Int32).alias("young_driver"),
    (pl.col("driver_age") > 70).cast(pl.Int32).alias("old_driver"),
])
df_age_pd = df_with_age.to_pandas()
df_age_pd["log_exposure"] = np.log(df_age_pd["exposure"].clip(lower=1e-6))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_freq_v2 = smf.glm(
        formula=(
            "claim_count ~ "
            "C(area) + "
            "C(ncd_years, Treatment(0)) + "
            "C(conviction_flag, Treatment(0)) + "
            "vehicle_group + "
            "young_driver + "
            "old_driver"
        ),
        data=df_age_pd,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df_age_pd["log_exposure"],
    ).fit()

print(f"v2 model converged: {glm_freq_v2.converged}")
```

**Task 1.** Extract relativities from both `glm_freq` (v1) and `glm_freq_v2` (v2). Build a comparison DataFrame with columns: `feature`, `level`, `v1_rel`, `v2_rel`, `pct_change`, and a `flag` column (True where the absolute percentage change exceeds 5%).

**Task 2.** Print the flagged rows. Which factors changed by more than 5%? Were you expecting this?

**Task 3.** For each flagged row, write one sentence explaining why the change occurred. Think about what happens to the area relativities when you add driver age to the model - if young drivers are more concentrated in some areas than others, what does that do to the area coefficients?

**Task 4.** Write a function `generate_change_log(v1_rels, v2_rels, threshold=0.05)` that produces a formatted text report suitable for presenting to the pricing committee. The report should include: model names, run date, total relativities compared, number flagged, and a table of all flagged factors. Then generate the report for the v1-to-v2 comparison.

---

### Solution - Exercise 4

```python
import polars as pl
import numpy as np
from datetime import date


def extract_rels_simple(glm_result, base_levels):
    """Simplified relativity extraction for comparison purposes."""
    params = glm_result.params
    records = []
    for param_name, coef in params.items():
        if param_name == "Intercept":
            continue
        if "[T." in param_name:
            feature_part = param_name.split("[T.")[0]
            level_part = param_name.split("[T.")[1].rstrip("]")
            if feature_part.startswith("C("):
                feature_part = feature_part[2:].split(",")[0].split(")")[0].strip()
        else:
            feature_part = param_name
            level_part = "continuous"
        records.append({
            "feature": feature_part,
            "level": level_part,
            "relativity": float(np.exp(coef)),
        })
    rels = pl.DataFrame(records)
    base_rows = pl.DataFrame([
        {"feature": f, "level": str(lv), "relativity": 1.0}
        for f, lv in base_levels.items()
    ])
    return pl.concat([base_rows, rels]).sort(["feature", "level"])


# Extract relativities from both models
v1_rels = extract_rels_simple(
    glm_freq,
    {"area": "A", "ncd_years": "0", "conviction_flag": "0"}
)
v2_rels = extract_rels_simple(
    glm_freq_v2,
    {"area": "A", "ncd_years": "0", "conviction_flag": "0"}
)

# Task 1: Comparison DataFrame
comparison = (
    v1_rels
    .join(v2_rels, on=["feature", "level"], how="inner", suffix="_v2")
    .rename({"relativity": "v1_rel", "relativity_v2": "v2_rel"})
    .with_columns(
        ((pl.col("v2_rel") / pl.col("v1_rel") - 1) * 100).alias("pct_change")
    )
    .with_columns(
        (pl.col("pct_change").abs() > 5.0).alias("flag")
    )
    .sort(["feature", "level"])
)

print("Task 1: Full comparison table")
print(comparison)
```

```python
# Task 2: Flagged rows
flagged = comparison.filter(pl.col("flag"))
print(f"\nTask 2: Flagged rows - relativity changes above 5%:")
if len(flagged) == 0:
    print("  None detected.")
else:
    print(flagged.select(["feature", "level", "v1_rel", "v2_rel", "pct_change"]))
```

**Task 3 explanation for flagged rows:**

When we add young_driver and old_driver flags to the model, those flags capture frequency variation that the v1 model was attributing (incorrectly) to other correlated factors. Young drivers tend to concentrate in certain areas and vehicle groups. In v1, the area coefficients were absorbing some of the young driver effect: areas with more young drivers had inflated area relativities. In v2, the age flags carry that effect directly, so those area relativities decrease. Similarly, NCD coefficients may shift if young drivers are more likely to have zero NCD. This is not an error in v1 - it is a property of omitted variable bias. The change log is evidence that v2 is more correctly specified, not that v1 was fundamentally wrong.

```python
# Task 4: Change log function
def generate_change_log(
    v1_rels: pl.DataFrame,
    v2_rels: pl.DataFrame,
    v1_name: str = "Model v1",
    v2_name: str = "Model v2",
    threshold: float = 0.05,
) -> str:
    """
    Generate a formatted change log for the pricing committee.

    Parameters
    ----------
    v1_rels, v2_rels : Polars DataFrames with columns feature, level, relativity
    v1_name, v2_name : human-readable model names
    threshold : flag any relativity that changes by more than this fraction (0.05 = 5%)

    Returns
    -------
    Formatted string report.
    """
    comparison = (
        v1_rels
        .join(v2_rels, on=["feature", "level"], how="inner", suffix="_v2")
        .rename({"relativity": "v1_rel", "relativity_v2": "v2_rel"})
        .with_columns(
            ((pl.col("v2_rel") / pl.col("v1_rel") - 1) * 100).alias("pct_change")
        )
        .with_columns(
            (pl.col("pct_change").abs() > threshold * 100).alias("flag")
        )
        .sort(["feature", "level"])
    )

    flagged = comparison.filter(pl.col("flag"))
    n_compared = len(comparison)
    n_flagged = len(flagged)

    lines = [
        "=" * 70,
        "PRICING MODEL CHANGE LOG",
        "=" * 70,
        f"Previous model:  {v1_name}",
        f"Current model:   {v2_name}",
        f"Report date:     {date.today().isoformat()}",
        f"Threshold:       {threshold * 100:.0f}% change",
        "",
        f"Relativities compared: {n_compared}",
        f"Flagged for review:    {n_flagged} ({n_flagged/n_compared*100:.1f}%)",
        "",
    ]

    if n_flagged == 0:
        lines.append("No material relativity changes detected.")
        lines.append("All changes are within the review threshold.")
    else:
        lines.append("FLAGGED CHANGES (require sign-off before deployment):")
        lines.append("-" * 70)
        lines.append(
            f"{'Factor':<22} {'Level':<10} {'Previous':>10} {'Current':>10} {'Change':>10}"
        )
        lines.append("-" * 70)
        for row in flagged.iter_rows(named=True):
            direction = "UP" if row["pct_change"] > 0 else "DOWN"
            lines.append(
                f"{row['feature']:<22} {str(row['level']):<10} "
                f"{row['v1_rel']:>10.4f} {row['v2_rel']:>10.4f} "
                f"{row['pct_change']:>+8.1f}% {direction}"
            )
        lines.append("-" * 70)
        lines.append("")
        lines.append("Action required: document reason for each flagged change.")
        lines.append("Obtain head of pricing sign-off before production deployment.")

    lines.append("=" * 70)
    return "\n".join(lines)


report = generate_change_log(
    v1_rels, v2_rels,
    v1_name="freq_glm_v1 (no age flags)",
    v2_name="freq_glm_v2 (with young/old driver flags)",
    threshold=0.05,
)
print(report)
```

**Discussion.** The change log function is the start of a proper model governance process. For a regulated insurer, this report should be:

- Generated automatically as part of the model deployment pipeline, not manually
- Stored in version control alongside the model code
- Attached to the pricing committee approval record
- Referenced in the FCA Consumer Duty product assessment documentation, and in your PS 21/5 pricing practices governance evidence

A change that is flagged but not documented is a regulatory risk. The FCA does not require that relativities stay constant - pricing should evolve as data improves. What they require is that changes are deliberate, documented, and shown to be appropriate for fair value. The automated change log makes that evidence-gathering part of the standard workflow rather than a retrospective exercise.
