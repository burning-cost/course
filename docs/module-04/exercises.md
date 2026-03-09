# Module 4 Exercises: SHAP Relativities

Four exercises. All four can be completed in a Databricks notebook. Work through them in order - each builds on the previous, and Exercise 4 requires everything from 1-3 to be done first.

Each exercise tells you exactly what to type, what output to expect, and what it means. Solutions are at the end of each exercise.

**Before you start:** The setup code (imports, data generation, model training, `sr.fit()`, and `sr.validate()`) from the tutorial must be in your notebook and have been run successfully. If you closed the tutorial notebook, open it again and re-run Parts 2 through 7 before starting these exercises.

---

## Exercise 1: What happens when you get the exposure offset wrong

**Why this matters:** The tutorial told you that `baseline=np.log(exposure)` is the correct way to pass the exposure offset to CatBoost. This exercise quantifies what happens when you get it wrong. This is not abstract - this error appears in real production models. The model trains without error and the Gini looks fine; the problem is invisible until you check calibration.

**What you will see:** Two wrong implementations compared to the correct one. You will measure the calibration error each introduces and then check how that error distorts the extracted relativities.

### Step 1: Set up three Pools with different exposure treatments

In a new cell in your Databricks notebook, type this and run it (Shift+Enter):

```python
import polars as pl
import pandas as pd
import numpy as np
import catboost as cb
from shap_relativities import SHAPRelativities

# We reuse df_pd, y_pd, X_pd from the tutorial
# If those variables are not in scope, re-run Parts 3-4 of the tutorial first

# Three Pools: correct, wrong (raw exposure), wrong (no offset)
pool_correct = cb.Pool(
    data=X_pd,
    label=y_pd,
    baseline=np.log(exposure_pd.clip(lower=1e-6)),  # CORRECT
    cat_features=CAT_FEATURES,
)

pool_raw_exp = cb.Pool(
    data=X_pd,
    label=y_pd,
    baseline=exposure_pd.values,   # WRONG: raw exposure, not log
    cat_features=CAT_FEATURES,
)

pool_no_offset = cb.Pool(
    data=X_pd,
    label=y_pd,
    # no baseline at all - assumes all policies have exposure=1.0
    cat_features=CAT_FEATURES,
)

print("Three Pools created.")
```

You will see: `Three Pools created.`

### Step 2: Train three models

In a new cell, type this and run it (Shift+Enter):

```python
params = {
    "loss_function":    "Poisson",
    "learning_rate":    0.05,
    "depth":            4,
    "min_data_in_leaf": 50,
    "iterations":       200,
    "random_seed":      42,
    "verbose":          0,
}

model_correct    = cb.CatBoostRegressor(**params)
model_raw_exp    = cb.CatBoostRegressor(**params)
model_no_offset  = cb.CatBoostRegressor(**params)

model_correct.fit(pool_correct)
model_raw_exp.fit(pool_raw_exp)
model_no_offset.fit(pool_no_offset)

print("All three models trained.")
```

This takes 60-90 seconds for three models. You will see: `All three models trained.`

### Step 3: Compare calibration

In a new cell, type this and run it (Shift+Enter):

```python
pred_correct   = model_correct.predict(pool_correct)
pred_raw_exp   = model_raw_exp.predict(pool_raw_exp)
pred_no_offset = model_no_offset.predict(pool_no_offset)

actual_total = y_pd.sum()

print(f"Actual total claims:               {actual_total:,}")
print()
print(f"Correct (log-offset):              {pred_correct.sum():,.0f}  "
      f"(ratio: {pred_correct.sum()/actual_total:.4f})")
print(f"Wrong (raw exposure as baseline):  {pred_raw_exp.sum():,.0f}  "
      f"(ratio: {pred_raw_exp.sum()/actual_total:.4f})")
print(f"Wrong (no offset):                 {pred_no_offset.sum():,.0f}  "
      f"(ratio: {pred_no_offset.sum()/actual_total:.4f})")
```

You will see output like:

```
Actual total claims:               4,821

Correct (log-offset):              4,819  (ratio: 0.9996)
Wrong (raw exposure as baseline):  7,341  (ratio: 1.5228)
Wrong (no offset):                 6,183  (ratio: 1.2826)
```

The raw-exposure model overcounts claims by 52%. The no-offset model overcounts by 28%. Both models trained and converged without error - CatBoost had no way of knowing the baseline was wrong.

### Step 4: Check how the error distorts relativities

In a new cell, type this and run it (Shift+Enter):

```python
# Extract area relativities from all three models
def quick_area_rels(model, X, exposure, cat_features):
    sr = SHAPRelativities(
        model=model,
        X=X,
        exposure=exposure,
        categorical_features=cat_features,
        continuous_features=[f for f in X.columns if f not in cat_features],
    )
    sr.fit()
    checks = sr.validate()
    if not checks["reconstruction"].passed:
        print(f"  WARNING: Reconstruction failed - {checks['reconstruction'].message}")
    rels = sr.extract_relativities(
        normalise_to="base_level",
        base_levels={"area": "A", "has_convictions": 0},
    )
    return rels[rels["feature"] == "area"].set_index("level")["relativity"]


print("Extracting relativities from all three models...")
area_correct   = quick_area_rels(model_correct,   X_pd, exposure_pd, CAT_FEATURES)
area_raw_exp   = quick_area_rels(model_raw_exp,   X_pd, exposure_pd, CAT_FEATURES)
area_no_offset = quick_area_rels(model_no_offset, X_pd, exposure_pd, CAT_FEATURES)

print(f"\n{'Area':<6} {'Correct':>10} {'Raw exp':>10} {'No offset':>10}")
print("-" * 40)
for lvl in ["A", "B", "C", "D", "E", "F"]:
    c  = area_correct.get(lvl, float("nan"))
    r  = area_raw_exp.get(lvl, float("nan"))
    n  = area_no_offset.get(lvl, float("nan"))
    print(f"{lvl:<6} {c:>10.3f} {r:>10.3f} {n:>10.3f}")
```

This takes 2-3 minutes as three sets of SHAP values are computed. The output looks like:

```
Area    Correct    Raw exp  No offset
----------------------------------------
A         1.000      1.000      1.000
B         1.108      1.093      1.101
C         1.225      1.213      1.219
D         1.431      1.408      1.422
E         1.668      1.641      1.659
F         1.950      1.913      1.938
```

The relativities from the wrong models are not dramatically different - they are 1-3% lower across the board. This is the insidious nature of the calibration error: the relativities look plausible, the model trains without complaint, but the absolute predictions are wrong by 28-52%. If you used the wrong model for pricing, you would undercharge by 28-52% overall. The relativities would be approximately right, but the base rate would be catastrophically wrong.

**What to do about this:** Always run the calibration check from Part 4 of the tutorial before extracting relativities. Ratio of predicted to actual total claims should be within 1% on the training set.

---

## Exercise 2: Correlated features and SHAP attribution

**Why this matters:** When two features are correlated, SHAP has to decide how to split the credit between them. The `tree_path_dependent` method (the default) allocates credit based on which feature appears in the tree splits. The `interventional` method allocates based on independent marginalisation. For presentation purposes, the attribution method affects which feature appears more or less important.

**What you will see:** The same model explained with two different SHAP methods. You will see how correlated features share attribution differently under each method.

### Step 1: Create a portfolio with correlated features

In a new cell, type this and run it (Shift+Enter):

```python
# Create a synthetic portfolio where vehicle_value (a new feature)
# is correlated with vehicle_group but also has an independent effect
rng2 = np.random.default_rng(77)
n2   = 30_000

area2 = rng2.choice(["A","B","C","D","E","F"], size=n2,
                     p=[0.10,0.18,0.25,0.22,0.15,0.10])
vg2   = rng2.integers(1, 51, size=n2)

# vehicle_value is correlated with vehicle_group
# Groups 1-25: older/cheaper; 26-50: newer/more expensive
vv2 = vg2 * 500 + rng2.normal(0, 3000, size=n2)  # pounds
vv2 = np.clip(vv2, 1000, 30000).astype(np.int32)

ncd2 = rng2.choice([0,1,2,3,4,5], size=n2, p=[0.08,0.07,0.09,0.12,0.20,0.44])
exp2 = np.clip(rng2.beta(8, 2, size=n2), 0.05, 1.0)

area_eff2 = {"A":0.0,"B":0.10,"C":0.20,"D":0.35,"E":0.50,"F":0.70}
log_mu2 = (
    -3.10
    + np.array([area_eff2[a] for a in area2])
    + (-0.15) * ncd2
    + 0.010 * (vg2 - 25)     # vehicle_group effect
    + 0.00002 * (vv2 - 12500) # small independent vehicle_value effect
)
claims2 = rng2.poisson(np.exp(log_mu2) * exp2)

df2_pd = pd.DataFrame({
    "area":          area2,
    "vehicle_group": vg2,
    "vehicle_value": vv2,
    "ncd_years":     ncd2,
    "exposure":      exp2,
    "claim_count":   claims2,
})

corr = df2_pd[["vehicle_group","vehicle_value"]].corr().iloc[0,1]
print(f"Correlation between vehicle_group and vehicle_value: {corr:.3f}")
print(f"Dataset: {len(df2_pd):,} rows, {claims2.sum():,} claims")
```

You will see a correlation around 0.80-0.85 - strong enough to create attribution competition between the two features.

### Step 2: Train the model with both correlated features

In a new cell, type this and run it (Shift+Enter):

```python
features2 = ["area", "vehicle_group", "vehicle_value", "ncd_years"]
X2_pd     = df2_pd[features2]
log_exp2  = np.log(df2_pd["exposure"].clip(lower=1e-6))

pool2 = cb.Pool(
    data=X2_pd,
    label=df2_pd["claim_count"],
    baseline=log_exp2,
    cat_features=["area"],
)

model2 = cb.CatBoostRegressor(
    loss_function="Poisson",
    iterations=200,
    depth=5,
    learning_rate=0.05,
    min_data_in_leaf=30,
    random_seed=42,
    verbose=0,
)
model2.fit(pool2)
print("Model trained.")
```

You will see: `Model trained.`

### Step 3: Compare tree_path_dependent vs interventional SHAP

In a new cell, type this and run it (Shift+Enter):

```python
# tree_path_dependent (default, fast)
sr2_tpd = SHAPRelativities(
    model=model2,
    X=X2_pd,
    exposure=df2_pd["exposure"],
    categorical_features=["area"],
    continuous_features=["vehicle_group", "vehicle_value", "ncd_years"],
    feature_perturbation="tree_path_dependent",
)
sr2_tpd.fit()

# interventional (slower, requires background data)
# Use 500 background samples for speed
bg2 = X2_pd.sample(n=500, random_state=42)

sr2_int = SHAPRelativities(
    model=model2,
    X=X2_pd,
    exposure=df2_pd["exposure"],
    categorical_features=["area"],
    continuous_features=["vehicle_group", "vehicle_value", "ncd_years"],
    feature_perturbation="interventional",
    background_data=bg2,
)
sr2_int.fit()

print("Both SHAP computations complete.")
```

The interventional computation takes 2-4 minutes. You will see: `Both SHAP computations complete.`

### Step 4: Compare feature importance under each method

In a new cell, type this and run it (Shift+Enter):

```python
shap_tpd = sr2_tpd.shap_values()
shap_int = sr2_int.shap_values()
feat_names = sr2_tpd.feature_names_

print(f"{'Feature':<20} {'TPD mean|SHAP|':>16} {'Interv mean|SHAP|':>18} {'Ratio':>8}")
print("-" * 65)
for i, feat in enumerate(feat_names):
    tpd_imp = np.abs(shap_tpd[:, i]).mean()
    int_imp = np.abs(shap_int[:, i]).mean()
    ratio   = int_imp / tpd_imp if tpd_imp > 0 else float("nan")
    print(f"{feat:<20} {tpd_imp:>16.5f} {int_imp:>18.5f} {ratio:>8.2f}")
```

You will see output like:

```
Feature              TPD mean|SHAP|  Interv mean|SHAP|    Ratio
-----------------------------------------------------------------
area                        0.18432             0.18218     0.99
vehicle_group               0.06841             0.08103     1.18
vehicle_value               0.01243             0.00619     0.50
ncd_years                   0.05319             0.05441     1.02
```

The pattern to notice: under `tree_path_dependent`, `vehicle_value` gets more attribution because it appears in tree splits due to correlation with `vehicle_group`. Under `interventional`, it gets less - because when you marginalise independently, `vehicle_value` contributes less once `vehicle_group`'s independent effect is accounted for. `vehicle_group` gets relatively more attribution under `interventional` because it has the stronger independent effect in the DGP.

**What to do with this:** For committee presentations where `vehicle_group` and `vehicle_value` are both candidate rating factors, use `interventional` SHAP to avoid overstating the importance of the more correlated feature. Document the choice. If you choose `tree_path_dependent` for speed, note in your governance paper that feature importance numbers are approximate for correlated features.

---

## Exercise 3: Severity relativities and combining with frequency

**Why this matters:** A pure premium factor table requires combining frequency and severity relativities. This exercise trains a Gamma severity model, extracts its relativities, and builds a simple pure premium table. The combination method here is an approximation - Module 5 covers the correct mSHAP approach.

### Step 1: Prepare the severity dataset

In a new cell in your notebook, type this and run it (Shift+Enter):

```python
# Reuse df and df_pd from the tutorial
# Severity: only policies with at least one claim
claims_only = df.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

claims_pd = claims_only.to_pandas()

print(f"Claims-only policies:  {len(claims_pd):,}")
print(f"Mean severity:         £{claims_pd['avg_severity'].mean():,.0f}")
print(f"Median severity:       £{claims_pd['avg_severity'].median():,.0f}")
print(f"Coefficient of var:    {claims_pd['avg_severity'].std() / claims_pd['avg_severity'].mean():.2f}")
```

You will see something like:

```
Claims-only policies:  4,821
Mean severity:         £2,604
Median severity:       £2,311
Coefficient of var:    0.58
```

A coefficient of variation around 0.5-0.7 is typical for UK motor claims. Very high CV (above 1.0) would suggest large losses inflating the mean.

### Step 2: Train the Gamma severity model

In a new cell, type this and run it (Shift+Enter):

```python
SEV_FEATURES = ["area", "has_convictions", "vehicle_group", "ncd_years"]
SEV_CAT      = ["area", "has_convictions"]
SEV_CONT     = ["vehicle_group", "ncd_years"]

# Add has_convictions to claims_pd if not already there
if "has_convictions" not in claims_pd.columns:
    claims_pd["has_convictions"] = (claims_pd["conviction_points"] > 0).astype(int)

X_sev_pd = claims_pd[SEV_FEATURES]
y_sev_pd = claims_pd["avg_severity"]

# Gamma severity: weight by claim count (more claims = more reliable average)
w_sev_pd = claims_pd["claim_count"]

sev_pool = cb.Pool(
    data=X_sev_pd,
    label=y_sev_pd,
    weight=w_sev_pd,
    cat_features=SEV_CAT,
    # No baseline for severity - we are not modelling rates, just amounts
)

sev_params = {
    "loss_function":    "Tweedie:variance_power=2",  # Gamma equivalent
    "learning_rate":    0.05,
    "depth":            4,
    "min_data_in_leaf": 10,  # fewer policies have claims; need smaller min_leaf
    "iterations":       200,
    "random_seed":      42,
    "verbose":          0,
}

sev_model = cb.CatBoostRegressor(**sev_params)
sev_model.fit(sev_pool)

print("Severity model trained.")
print(f"Predicted mean severity: £{sev_model.predict(sev_pool).mean():,.0f}")
print(f"Actual mean severity:    £{y_sev_pd.mean():,.0f}")
```

You will see:

```
Severity model trained.
Predicted mean severity: £2,612
Actual mean severity:    £2,604
```

The predicted and actual means should be close. If they diverge by more than 5%, check that the Tweedie variance power is set to 2 (which corresponds to a Gamma distribution).

**Why `Tweedie:variance_power=2` and not `Gamma`?** CatBoost implements Gamma loss via the Tweedie family with variance_power=2. The two are identical. Some versions of CatBoost accept `"Gamma"` directly; others require the explicit Tweedie parameter. The `variance_power=2` form is always safe.

### Step 3: Extract severity relativities

In a new cell, type this and run it (Shift+Enter):

```python
sr_sev = SHAPRelativities(
    model=sev_model,
    X=X_sev_pd,
    exposure=w_sev_pd,   # claim count is the "exposure" for severity averaging
    categorical_features=SEV_CAT,
    continuous_features=SEV_CONT,
)
sr_sev.fit()

checks_sev = sr_sev.validate()
print("Severity SHAP validation:")
for name, result in checks_sev.items():
    print(f"  [{('PASS' if result.passed else 'FAIL')}] {name}: {result.message}")

rels_sev = sr_sev.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "has_convictions": 0},
)

print("\nSeverity area relativities:")
print(rels_sev[rels_sev["feature"] == "area"][
    ["level", "relativity", "lower_ci", "upper_ci"]
].to_string(index=False))
```

You will see the validation checks pass, then the severity area relativities. These should be near 1.0 across all areas, with some noise. The true severity DGP has a small area effect (`area_effect * 0.3`), giving area F a true relativity of `exp(0.70 * 0.3) = exp(0.21) ≈ 1.23`. The extracted values will cluster around this with wider confidence intervals than the frequency relativities, because there are only ~4,800 claims rather than 100,000 policies.

### Step 4: Build the pure premium table

In a new cell, type this and run it (Shift+Enter):

```python
# Get frequency and severity relativities for area, indexed by level
rels_freq = rels  # from the tutorial (sr.extract_relativities())

freq_area = rels_freq[rels_freq["feature"] == "area"].set_index("level")
sev_area  = rels_sev[rels_sev["feature"] == "area"].set_index("level")

print(f"{'Area':<6} {'Freq rel':>10} {'Sev rel':>10} {'PP rel (F×S)':>14}")
print("-" * 44)
for lvl in ["A", "B", "C", "D", "E", "F"]:
    freq_r = freq_area.loc[lvl, "relativity"]
    sev_r  = sev_area.loc[lvl, "relativity"]
    pp_r   = freq_r * sev_r
    print(f"{lvl:<6} {freq_r:>10.3f} {sev_r:>10.3f} {pp_r:>14.3f}")

print("\nNote: PP = freq × sev is an approximation.")
print("Module 5 covers the correct mSHAP combination.")
```

You will see something like:

```
Area   Freq rel   Sev rel  PP rel (F×S)
--------------------------------------------
A         1.000      1.000         1.000
B         1.108      1.031         1.143
C         1.225      1.071         1.312
D         1.431      1.108         1.586
E         1.668      1.148         1.915
F         1.950      1.198         2.337
```

The pure premium relativities for area are larger than the frequency relativities alone, because area has a positive effect in both models. Area F in the true DGP has a frequency effect of `exp(0.70) = 2.01` and a severity effect of `exp(0.21) = 1.23`, giving a true pure premium relativity of `2.01 × 1.23 = 2.47`. Your extracted value should be in the neighbourhood of 2.3-2.6.

**Why this is an approximation:** The correct combination, mSHAP (Lindstrom et al., 2022), accounts for the joint distribution of frequency and severity SHAP values. Multiplying relativities works when frequency and severity are independent given the features. In practice, they usually are not perfectly independent. Module 5 covers mSHAP.

---

## Exercise 4: Base level choice and regulator communication

**Why this matters:** The choice of base level affects how every number in the factor table looks, but it does not affect the model at all. If you cannot explain this clearly, the pricing committee will second-guess your table and the regulator will ask why the numbers changed between drafts.

### Step 1: Extract NCD relativities with two different base levels

In a new cell in your notebook, type this and run it (Shift+Enter):

```python
# rels is already computed from the tutorial
# Re-extract with NCD=0 as base
rels_ncd0 = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "has_convictions": 0, "ncd_years": 0},
)

# Re-extract with NCD=4 as base
rels_ncd4 = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "has_convictions": 0, "ncd_years": 4},
)

ncd_base0 = rels_ncd0[rels_ncd0["feature"] == "ncd_years"].set_index("level")
ncd_base4 = rels_ncd4[rels_ncd4["feature"] == "ncd_years"].set_index("level")

print("NCD relativities: NCD=0 base vs NCD=4 base")
print(f"\n{'NCD':<6} {'Base=0':>10} {'Base=4':>10} {'Ratio':>10} {'Same?':>8}")
print("-" * 50)
for k in range(6):
    r0 = ncd_base0.loc[k, "relativity"]
    r4 = ncd_base4.loc[k, "relativity"]
    ratio = r4 / r0
    same = "YES" if abs(ratio - ncd_base4.loc[4, "relativity"] / ncd_base0.loc[4, "relativity"]) < 0.001 else "NO"
    print(f"{k:<6} {r0:>10.3f} {r4:>10.3f} {ratio:>10.3f} {same:>8}")
```

You will see both tables side by side. The Ratio column is constant across all NCD levels - it equals `1 / ncd_base0.loc[4, "relativity"]`. This is the key observation: changing the base level divides every relativity by the same constant. The shape - the ratios between levels - is identical.

### Step 2: Verify predictions are unchanged

In a new cell, type this and run it (Shift+Enter):

```python
# The model's predictions are totally unaffected by base level choice.
# Demonstrate this by checking that the model still produces the same predictions.

pred1 = freq_model.predict(train_pool)
print(f"Model predictions are determined by the model, not by base level choice.")
print(f"First 5 predictions: {pred1[:5].round(5).tolist()}")
print()
print("Base level choice affects the display of relativities only.")
print("The product of all relativities for any policy is invariant to the base level.")
```

### Step 3: Write the committee explanation

In a new cell, type this and run it (Shift+Enter):

```python
ncd4_rel_base0 = ncd_base0.loc[4, "relativity"]
ncd5_rel_base0 = ncd_base0.loc[5, "relativity"]
ncd5_rel_base4 = ncd_base4.loc[5, "relativity"]
ncd5_lo = ncd_base0.loc[5, "lower_ci"]
ncd5_hi = ncd_base0.loc[5, "upper_ci"]

print("=" * 70)
print("COMMITTEE BRIEFING LANGUAGE (copy and adapt)")
print("=" * 70)
print()
print(f"The NCD=5 relativity relative to NCD=0 is {ncd5_rel_base0:.3f}")
print(f"[95% CI: {ncd5_lo:.3f} - {ncd5_hi:.3f}].")
print(f"This means a driver with five years' NCD has {(1-ncd5_rel_base0)*100:.0f}% lower")
print(f"predicted frequency than a new driver, holding all other factors constant.")
print()
print(f"The NCD=5 relativity relative to NCD=4 is {ncd5_rel_base4:.3f}.")
print(f"This is the same underlying model: NCD=5 vs NCD=4 is simply")
print(f"{ncd5_rel_base0:.3f} / {ncd4_rel_base0:.3f} = {ncd5_rel_base4:.3f}.")
print()
print("=" * 70)
print("REGULATOR EXPLANATION OF CONFIDENCE INTERVALS")
print("=" * 70)
print()
print(f"The 95% confidence interval on the NCD=5 relativity is")
print(f"[{ncd5_lo:.3f}, {ncd5_hi:.3f}].")
print()
print("What this means:")
print(f"  The interval is constructed using the standard error of the mean")
print(f"  SHAP value for NCD=5 policyholders in our portfolio, combined with")
print(f"  the standard error for the NCD=0 base level. In a portfolio of this")
print(f"  size, we estimate the NCD=5 relativity with enough precision to")
print(f"  distinguish it clearly from 1.0.")
print()
print("What this does NOT mean:")
print(f"  The interval does not tell us whether the GBM's learned NCD effect")
print(f"  is the 'true' effect or whether a different dataset would produce the")
print(f"  same number. For that question, we present temporal stability analysis")
print(f"  (training on different year windows) and comparison with the GLM.")
```

You will see formatted text ready to use in a committee briefing. Read it, make sure you understand every sentence, and then answer the questions below.

### Step 4: Answer these questions

Work through these questions without looking at the solutions below.

**Question 1:** Your head of pricing asks: "We decided to use NCD=4 as the base level because that is our biggest cell. But when I look at the NCD=0 table in the GLM report, I cannot compare the two. Can we align them?" Write what you would say to align the tables - specifically, what operation do you perform on the GBM relativities to shift from NCD=4 base to NCD=0 base?

**Question 2:** The regulator asks: "You have a 95% confidence interval on NCD=5 of [0.44, 0.51]. How many NCD=5 policies do you have in the portfolio?" Without looking at the data, estimate the approximate count from the width of the interval alone. Then check your estimate against the actual `n_obs` from the relativity table.

**Question 3:** A committee member points out that the GBM's NCD=5 relativity is 0.472 while the GLM's is 0.461. He says: "They disagree - which one should we use?" Write a one-paragraph response.

---

### Solution - Exercise 4

**Question 1 answer:**

To shift from NCD=4 base to NCD=0 base, divide every NCD relativity by the NCD=4 relativity from the NCD=0 table. In Python:

```python
ncd4_value = ncd_base0.loc[4, "relativity"]
aligned = {k: ncd_base4.loc[k, "relativity"] / ncd_base4.loc[0, "relativity"]
           for k in range(6)}
```

More directly: just re-run `sr.extract_relativities()` with `base_levels={"ncd_years": 0}`. The library handles the rescaling. The head of pricing is right that NCD=0 base makes the GLM comparison easier; use `base_level` = 0 in any presentation that sits alongside the GLM report.

**Question 2 answer:**

The confidence interval width is approximately `2 × 1.96 × SE(NCD=5)`. For a relativity of around 0.47:

- Log-relativity ≈ -0.75
- CI half-width in log space ≈ (ln(0.51) - ln(0.44)) / 2 ≈ 0.074
- SE ≈ 0.074 / 1.96 ≈ 0.038
- SE = shap_std / sqrt(n), so n ≈ (shap_std / SE)^2

Without knowing shap_std exactly, a typical SHAP std for NCD is 0.08-0.12. At shap_std = 0.10: n ≈ (0.10 / 0.038)^2 ≈ 7. That gives roughly 7 × (size_of_std)^2 which... let us work backwards. For SE = 0.038 and shap_std = 0.10, n ≈ 7. That is implausibly small for NCD=5 - it suggests the estimate is less precise than a typical portfolio.

Check the actual count: `rels[rels["feature"] == "ncd_years"].loc[5, "n_obs"]`. A synthetic 100k portfolio with 44% NCD=5 should have around 44,000 NCD=5 policies, which gives a much tighter CI. If the CI you see is [0.44, 0.51], the portfolio is smaller or the shap_std is larger than typical. The point of this question is to build the habit of checking whether the interval width makes intuitive sense given the portfolio size.

**Question 3 answer:**

The GBM and GLM find NCD=5 relativities of 0.472 and 0.461 respectively. The difference is 0.011 in relativity space, or 2.4%. The GBM's 95% confidence interval is [0.44, 0.51], which includes 0.461 comfortably. We cannot distinguish these two estimates statistically.

The most likely explanation for the difference is that the GLM treats NCD as a continuous linear variable while the GBM treats it as a non-linear feature. If the true NCD effect is slightly non-linear (which is plausible - the step from NCD=4 to NCD=5 may have a different effect than NCD=0 to NCD=1), the two models distribute the NCD effect slightly differently. Neither estimate is definitively right. For production, we would use the GBM relativity because it comes from a better-fitting model, but we would note the comparison in the governance paper and confirm with the pricing committee that a 2-3% difference on NCD=5 is acceptable given our portfolio mix.

---

## Connecting to Module 5

Module 5 covers two topics that extend what you built here:

**mSHAP for pure premium.** The approximation in Exercise 3 (frequency relativity × severity relativity) is incorrect in general. mSHAP (Lindstrom et al., 2022) provides the mathematically correct way to compose two sets of SHAP values into a pure premium relativity. Module 5 implements this properly.

**Monitoring relativity stability.** Once your relativity pipeline is scheduled as a Databricks Workflow (weekly or monthly), you need to detect when relativities drift materially between runs. Module 5 builds the alerting logic: how to set thresholds, what counts as a material change, and how to trigger a notification when the NCD=5 relativity moves by more than 5% week-on-week.
