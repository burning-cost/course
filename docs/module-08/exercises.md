# Module 8 Exercises: End-to-End Pricing Pipeline

These exercises work through the pipeline from a different angle to the tutorial: rather than building the happy path, they explore what happens when stages are connected incorrectly, when inputs change between runs, and when the pipeline fails. By the end, you will have built a working pipeline from scratch on your own synthetic data and debugged each stage independently.

The exercises build on each other. Exercise 1 must be completed before Exercise 2. Exercises 5 and 6 are independent and can be done in any order after Exercise 4.

---

## Exercise 1: The feature engineering trap -- build it once, use it everywhere

**References:** Tutorial Parts 7 and 15.

**What this exercise covers:** The most common production failure in ML-based pricing is feature engineering defined in two places. This exercise makes you experience the failure before showing you the correct pattern.

**Setup.** The following cell generates a small synthetic dataset. Run it first.

```python
import polars as pl
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

rng = np.random.default_rng(seed=42)
n   = 15_000

df = pl.DataFrame({
    "ncb_years":      rng.choice([0,1,2,3,4,5], n, p=[0.08,0.07,0.10,0.15,0.20,0.40]).tolist(),
    "vehicle_group":  rng.integers(1, 51, n).tolist(),
    "region":         rng.choice(["North","Midlands","London","SouthEast","SouthWest"], n).tolist(),
    "driver_age":     rng.integers(17, 85, n).tolist(),
    "exposure":       np.clip(rng.beta(8, 2, n), 0.05, 1.0).tolist(),
    "claim_count":    rng.poisson(0.07, n).tolist(),
    "accident_year":  rng.choice([2021,2022,2023,2024], n).tolist(),
})

print(f"Dataset: {df.shape[0]:,} rows")
print(df.head(5))
```

### Task 1: Reproduce the mismatch failure

The training pipeline below encodes `ncb_years` as a string categorical before passing it to CatBoost. The scoring pipeline passes `ncb_years` as an integer.

1a. Run the training pipeline exactly as shown. Fit a CatBoost Poisson model. Record its MLflow run ID.

```python
# TRAINING PIPELINE
# Note: ncb_years cast to string before Pool construction
FEATURES_TRAIN  = ["ncb_years", "vehicle_group", "region", "driver_age"]
CAT_TRAIN       = ["region", "ncb_years"]   # ncb_years is categorical in training

df_train = df.filter(pl.col("accident_year") < 2024)
df_test  = df.filter(pl.col("accident_year") == 2024)

# Training encodes ncb_years as Utf8
df_train_enc = df_train.with_columns(pl.col("ncb_years").cast(pl.Utf8))
df_test_enc  = df_test.with_columns(pl.col("ncb_years").cast(pl.Utf8))

X_tr = df_train_enc[FEATURES_TRAIN].to_pandas()
y_tr = df_train["claim_count"].to_numpy()
w_tr = df_train["exposure"].to_numpy()

X_te = df_test_enc[FEATURES_TRAIN].to_pandas()
y_te = df_test["claim_count"].to_numpy()
w_te = df_test["exposure"].to_numpy()

train_pool = Pool(X_tr, y_tr, baseline=np.log(np.clip(w_tr, 1e-6, None)),
                  cat_features=CAT_TRAIN)
test_pool  = Pool(X_te, y_te, baseline=np.log(np.clip(w_te, 1e-6, None)),
                  cat_features=CAT_TRAIN)

model_correct = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=5, random_seed=42, verbose=0
)
model_correct.fit(train_pool, eval_set=test_pool)
print("Training complete (correct encoding).")
```

1b. Now simulate the scoring pipeline mismatch: pass `ncb_years` as an integer, not a string.

```python
# SCORING PIPELINE (wrong: ncb_years as integer)
X_te_wrong = df_test[FEATURES_TRAIN].to_pandas()   # ncb_years NOT cast to string
w_te_wrong  = df_test["exposure"].to_numpy()

# Note: cat_features only contains "region" -- ncb_years is now treated as continuous
wrong_pool = Pool(X_te_wrong, cat_features=["region"])
pred_wrong = model_correct.predict(wrong_pool)

# Correct scoring
pred_correct = model_correct.predict(test_pool)

# Compute the prediction error
mae = np.mean(np.abs(pred_correct - pred_wrong))
print(f"Mean absolute prediction error from encoding mismatch: {mae:.4f} claims")
print(f"As % of mean predicted count: {mae / pred_correct.mean():.1%}")
print(f"Was an exception raised? No -- this is a silent failure.")
```

**Question:** Did CatBoost raise an exception? What is the magnitude of the prediction error? Is this error large enough to matter for pricing?

### Task 2: Build the FeatureSpec guard

Implement a `FeatureSpec` class (or use the one from Tutorial Part 15) that:
- Records the expected dtype and unique values for each feature at training time
- Raises a `ValueError` if the scoring data does not match the spec
- Logs the spec JSON as an MLflow artefact alongside the trained model

Requirements:
- The spec must catch the mismatch in Task 1 before any predictions are made
- The error message must tell the user exactly which column is wrong and what to do about it
- The spec must be serialisable to JSON and reloadable from JSON

```python
import json
import mlflow

class FeatureSpec:
    def __init__(self):
        self.spec = {}

    def record(self, df: pl.DataFrame, cat_features: list) -> None:
        # Your implementation here
        pass

    def validate(self, df: pl.DataFrame) -> list:
        # Your implementation here
        # Return list of error strings. Empty = pass.
        pass

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FeatureSpec":
        obj = cls()
        with open(path) as f:
            obj.spec = json.load(f)
        return obj

# Test it: record on training data, validate on the wrong scoring data from Task 1
spec = FeatureSpec()
spec.record(df_train_enc.select(FEATURES_TRAIN), cat_features=CAT_TRAIN)
spec.to_json("/tmp/feature_spec_ex1.json")

spec_loaded = FeatureSpec.from_json("/tmp/feature_spec_ex1.json")

# This should fail loudly with a clear error
errors = spec_loaded.validate(pl.from_pandas(X_te_wrong))
print(f"Errors found: {len(errors)}")
for err in errors:
    print(f"  ERROR: {err}")
```

### Task 3: The canonical fix

Write a `build_feature_matrix()` function that encodes all features correctly and is called identically at training time and scoring time. Show that calling this function at both stages eliminates the mismatch from Task 1.

```python
def build_feature_matrix(
    df: pl.DataFrame,
    features: list,
    cat_features: list,
) -> tuple:
    """
    Single source of truth for feature engineering.
    Returns: (X_pandas, cat_features_list)

    Rules:
    - ncb_years must always be cast to Utf8 (categorical)
    - All other encoding must happen inside this function
    - Never call this function with pre-encoded data
    """
    # Your implementation here
    pass

# Verify: training and scoring produce the same feature dtypes
X_tr_canonical, cats = build_feature_matrix(df_train, FEATURES_TRAIN, CAT_TRAIN)
X_te_canonical, _    = build_feature_matrix(df_test,  FEATURES_TRAIN, CAT_TRAIN)

print("Training feature dtypes:")
print(X_tr_canonical.dtypes)
print("\nScoring feature dtypes:")
print(X_te_canonical.dtypes)
print("\nDtype match:", X_tr_canonical.dtypes.to_dict() == X_te_canonical.dtypes.to_dict())
```

**Expected output:** Both dtype dictionaries should be identical.

<details>
<summary>Solution -- Task 2 FeatureSpec</summary>

```python
class FeatureSpec:
    def __init__(self):
        self.spec = {}

    def record(self, df: pl.DataFrame, cat_features: list) -> None:
        for col in df.columns:
            series = df[col]
            if col in cat_features or series.dtype in (pl.Utf8, pl.Categorical):
                self.spec[col] = {
                    "dtype":       "categorical",
                    "unique_vals": sorted(
                        [str(v) for v in series.drop_nulls().unique().to_list()]
                    ),
                }
            else:
                self.spec[col] = {
                    "dtype": "numeric",
                    "min":   float(series.min()),
                    "max":   float(series.max()),
                }

    def validate(self, df: pl.DataFrame) -> list:
        errors = []
        for col, col_spec in self.spec.items():
            if col not in df.columns:
                errors.append(
                    f"Missing column '{col}'. Required columns: {list(self.spec.keys())}"
                )
                continue
            series = df[col]
            if col_spec["dtype"] == "categorical":
                if series.dtype not in (pl.Utf8, pl.Categorical, pl.Enum):
                    errors.append(
                        f"Column '{col}': expected categorical dtype (pl.Utf8 or pl.Categorical), "
                        f"got {series.dtype}. Cast with: pl.col('{col}').cast(pl.Utf8)"
                    )
            else:  # numeric
                if series.dtype in (pl.Utf8, pl.Categorical, pl.Enum):
                    errors.append(
                        f"Column '{col}': expected numeric dtype, got {series.dtype}."
                    )
        return errors
```

</details>

<details>
<summary>Solution -- Task 3 canonical function</summary>

```python
def build_feature_matrix(
    df: pl.DataFrame,
    features: list,
    cat_features: list,
) -> tuple:
    # Apply all categorical casts
    df_enc = df.with_columns([
        pl.col(col).cast(pl.Utf8) if col in cat_features else pl.col(col)
        for col in features
    ])
    X = df_enc.select(features).to_pandas()
    return X, cat_features
```

</details>

---

## Exercise 2: The Poisson offset -- weight vs baseline

**References:** Tutorial Parts 2 and 8.

**What this exercise covers:** The difference between `weight=exposure` and `baseline=np.log(exposure)` in CatBoost. This is the blocking issue from the head of pricing review.

**Setup.** Generate a simple dataset where you know the true claim frequency.

```python
rng2 = np.random.default_rng(seed=100)
n2   = 20_000

exposure2     = rng2.uniform(0.1, 1.0, n2)
true_freq2    = 0.08 * np.exp(0.3 * (exposure2 - 0.55))   # mild exposure effect
claim_count2  = rng2.poisson(true_freq2 * exposure2)

df2 = pd.DataFrame({
    "exposure":    exposure2,
    "claim_count": claim_count2,
    "feature":     rng2.normal(0, 1, n2),
})

n_train2, n_test2 = int(n2 * 0.8), int(n2 * 0.2)
X_tr2 = df2[["feature", "exposure"]].iloc[:n_train2]
y_tr2 = df2["claim_count"].values[:n_train2]
w_tr2 = df2["exposure"].values[:n_train2]

X_te2 = df2[["feature"]].iloc[n_train2:]
y_te2 = df2["claim_count"].values[n_train2:]
w_te2 = df2["exposure"].values[n_train2:]

print(f"True mean frequency: {true_freq2.mean():.4f}")
print(f"Observed mean frequency: {(claim_count2.sum() / exposure2.sum()):.4f}")
```

### Task 1: Fit both versions

Fit two models:
- Model A: `weight=w_tr2` (wrong -- this is the Module 8 bug)
- Model B: `baseline=np.log(w_tr2)` (correct -- exposure offset)

Both models should use only `"feature"` as the input feature, with `"exposure"` handled via the Pool parameter.

```python
# Model A: wrong offset (weight, not baseline)
pool_A_train = Pool(X_tr2[["feature"]], y_tr2, weight=w_tr2)
pool_A_test  = Pool(X_te2,              y_te2, weight=w_te2)

model_A = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4, random_seed=42, verbose=0
)
model_A.fit(pool_A_train, eval_set=pool_A_test)

# Model B: correct offset (baseline=log(exposure))
pool_B_train = Pool(X_tr2[["feature"]], y_tr2,
                    baseline=np.log(np.clip(w_tr2, 1e-6, None)))
pool_B_test  = Pool(X_te2,              y_te2,
                    baseline=np.log(np.clip(w_te2, 1e-6, None)))

model_B = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4, random_seed=42, verbose=0
)
model_B.fit(pool_B_train, eval_set=pool_B_test)
```

### Task 2: Compare predictions

For a policy with exposure = 1.0 year, Model A and Model B should give the same prediction. For exposure = 0.5 years, they should differ. Why?

```python
# Evaluate both models on the test set
# For Model A: predictions are on the count scale (no exposure offset in output)
pred_A = model_A.predict(Pool(X_te2))

# For Model B: predictions include the exposure offset
pred_B = model_B.predict(pool_B_test)

# What does each model predict for a full-year policy (exposure=1.0)?
# What does it predict for a half-year policy (exposure=0.5)?
# Your analysis here:

print("Predicted claims, full-year policies (exposure=1.0 in test):")
full_year_mask = (w_te2 > 0.95) & (w_te2 < 1.05)
print(f"  Model A (wrong): mean = {pred_A[full_year_mask].mean():.4f}")
print(f"  Model B (correct): mean = {pred_B[full_year_mask].mean():.4f}")

print("\nPredicted claims, half-year policies (exposure ~0.5):")
half_year_mask = (w_te2 > 0.45) & (w_te2 < 0.55)
print(f"  Model A (wrong): mean = {pred_A[half_year_mask].mean():.4f}")
print(f"  Model B (correct): mean = {pred_B[half_year_mask].mean():.4f}")

print("\nExpected ratio (half-year should predict roughly half the count):")
print(f"  Model A ratio: {pred_A[half_year_mask].mean() / pred_A[full_year_mask].mean():.3f}")
print(f"  Model B ratio: {pred_B[half_year_mask].mean() / pred_B[full_year_mask].mean():.3f}")
```

### Task 3: Compute the Poisson deviance for both models

Use the `poisson_deviance()` function from Tutorial Part 8. Which model achieves lower deviance on the test set?

```python
def poisson_deviance(y_true, y_pred, exposure):
    fp = np.clip(y_pred / exposure, 1e-10, None)
    ft = y_true / exposure
    d  = 2 * exposure * (np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp))
    return float(d.sum() / exposure.sum())

dev_A = poisson_deviance(y_te2, pred_A, w_te2)
dev_B = poisson_deviance(y_te2, pred_B, w_te2)

print(f"Model A (weight=exposure) Poisson deviance: {dev_A:.5f}")
print(f"Model B (baseline=log_exp) Poisson deviance: {dev_B:.5f}")
print(f"Relative difference: {(dev_A - dev_B) / dev_B:.1%}")
```

**Expected finding:** Model B should achieve lower Poisson deviance because it correctly models the exposure structure. Model A produces the same prediction for a 0.5-year policy and a 1.0-year policy with the same feature values -- it treats both as if they have the same expected claim count, which is wrong.

### Task 4: Explain to a non-technical colleague

Write a three-sentence explanation of the difference between `weight` and `baseline` in CatBoost that would be understood by a pricing actuary who has never used gradient boosting. Your answer should cover: what each parameter does, which is correct for a Poisson frequency model, and what goes wrong when you use the wrong one.

---

## Exercise 3: Walk-forward CV -- understanding temporal structure

**References:** Tutorial Part 8.

**What this exercise covers:** Walk-forward cross-validation and why a random split produces misleading validation metrics.

**Setup.**

```python
import polars as pl
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

rng3 = np.random.default_rng(seed=333)
n3   = 60_000

# Simulate a portfolio with a genuine time trend:
# claim frequency increases by 3% per year (loss cost trend)
accident_year3 = rng3.choice([2020,2021,2022,2023,2024], n3)
time_trend     = 0.03 * (accident_year3 - 2020)   # 0 in 2020, 0.12 in 2024

ncb3     = rng3.choice([0,1,2,3,4,5], n3)
exposure3 = np.clip(rng3.beta(8, 2, n3), 0.1, 1.0)
true_freq3 = np.exp(-2.9 - 0.18*ncb3 + time_trend)
claims3   = rng3.poisson(true_freq3 * exposure3)

df3 = pl.DataFrame({
    "accident_year": accident_year3.tolist(),
    "ncb_years":     ncb3.tolist(),
    "exposure":      exposure3.tolist(),
    "claim_count":   claims3.tolist(),
})

print("Observed claim frequency by accident year:")
print(df3.group_by("accident_year").agg(
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
).sort("accident_year").with_columns(
    (pl.col("claims") / pl.col("exposure")).round(5).alias("freq")
))
```

### Task 1: Random split vs temporal split -- compare validation metrics

Fit the same CatBoost Poisson model using two different validation strategies:
- Random 80/20 split (wrong for insurance)
- Temporal split: train on 2020-2023, test on 2024 (correct)

Use the Poisson deviance as the evaluation metric. Report the validation deviance for each approach.

```python
FEATURES3 = ["ncb_years"]

df3_pd = df3.to_pandas()

# --- Random split ---
X_rand, y_rand, w_rand = (
    df3_pd[FEATURES3], df3_pd["claim_count"].values, df3_pd["exposure"].values
)
X_rand_tr, X_rand_te, y_rand_tr, y_rand_te, w_rand_tr, w_rand_te = train_test_split(
    X_rand, y_rand, w_rand, test_size=0.2, random_state=42
)

pool_rand_tr = Pool(X_rand_tr, y_rand_tr,
                    baseline=np.log(np.clip(w_rand_tr, 1e-6, None)))
pool_rand_te = Pool(X_rand_te, y_rand_te,
                    baseline=np.log(np.clip(w_rand_te, 1e-6, None)))

model_rand = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4, random_seed=42, verbose=0
)
model_rand.fit(pool_rand_tr, eval_set=pool_rand_te)
pred_rand = model_rand.predict(pool_rand_te)
dev_rand  = poisson_deviance(y_rand_te, pred_rand, w_rand_te)

# --- Temporal split ---
mask_tr3 = df3_pd["accident_year"] < 2024
mask_te3 = df3_pd["accident_year"] == 2024

X_temp_tr = df3_pd.loc[mask_tr3, FEATURES3]
y_temp_tr = df3_pd.loc[mask_tr3, "claim_count"].values
w_temp_tr = df3_pd.loc[mask_tr3, "exposure"].values

X_temp_te = df3_pd.loc[mask_te3, FEATURES3]
y_temp_te = df3_pd.loc[mask_te3, "claim_count"].values
w_temp_te = df3_pd.loc[mask_te3, "exposure"].values

pool_temp_tr = Pool(X_temp_tr, y_temp_tr,
                    baseline=np.log(np.clip(w_temp_tr, 1e-6, None)))
pool_temp_te = Pool(X_temp_te, y_temp_te,
                    baseline=np.log(np.clip(w_temp_te, 1e-6, None)))

model_temp = CatBoostRegressor(
    loss_function="Poisson", iterations=200, depth=4, random_seed=42, verbose=0
)
model_temp.fit(pool_temp_tr, eval_set=pool_temp_te)
pred_temp = model_temp.predict(pool_temp_te)
dev_temp  = poisson_deviance(y_temp_te, pred_temp, w_temp_te)

print(f"Random split validation deviance:   {dev_rand:.5f}")
print(f"Temporal split validation deviance: {dev_temp:.5f}")
print(f"\nThe random split deviance is {'lower' if dev_rand < dev_temp else 'higher'}, "
      f"by {abs(dev_rand - dev_temp):.5f}.")
print("If random split deviance is lower, the model appears to perform better")
print("than it actually does on future data. This is the validation inflation bias.")
```

### Task 2: Measure the bias in practice

The random split includes 2024 observations in training (20% of 2024 rows appear in the training set in a random split). This means the model has seen some 2024 data during training, which makes validation on the remaining 2024 rows look better than it really is.

Quantify this bias: what is the difference between the validation deviance under each approach? Is it large enough to matter for a pricing decision?

### Task 3: IBNR buffer

Add an IBNR buffer to the temporal split: remove the last three months of each training year from the training set (simulate by removing 25% of each training year's rows, sampling from those with the highest policy IDs as a proxy for late-year policies).

Does the IBNR buffer change the validation deviance materially on this synthetic data? Why or why not?

---

## Exercise 4: Severity prediction -- all policies, not just claims

**References:** Tutorial Part 10 (Stage 6, "Computing pure premiums").

**What this exercise covers:** The severity imputation bug: filling zero-claim policies with the claims-only mean severity overestimates pure premium for low-risk policies.

**Setup.**

```python
rng4 = np.random.default_rng(seed=444)
n4   = 30_000

# Risk factor: low risk (risk=0) has lower frequency AND lower severity
# High risk (risk=1) has higher frequency AND higher severity
risk4      = rng4.choice([0, 1], n4, p=[0.70, 0.30])
exposure4  = np.clip(rng4.beta(8, 2, n4), 0.1, 1.0)
true_freq4 = np.where(risk4 == 0, 0.04, 0.14)   # low risk = 4%, high risk = 14%
claims4    = rng4.poisson(true_freq4 * exposure4)

# Severity: low risk has lower mean severity
true_mean_sev4 = np.where(risk4 == 0, 1_800, 4_200)
severity4 = np.where(
    claims4 > 0,
    rng4.lognormal(np.log(true_mean_sev4), 0.8, n4),
    0.0
)
incurred4 = claims4 * severity4 / np.maximum(claims4, 1)  # mean severity per claim
incurred4 = np.where(claims4 > 0, incurred4, 0.0)

FEATURES4 = ["risk"]
df4 = pd.DataFrame({
    "risk":         risk4, "exposure": exposure4,
    "claim_count":  claims4, "incurred_loss": severity4 * claims4,
})

print(f"Zero-claim policies: {(claims4 == 0).sum():,}  ({(claims4 == 0).mean():.1%})")
print(f"Claims-only mean severity: £{severity4[claims4 > 0].mean():,.0f}")
print(f"True mean severity (all risks): £{true_mean_sev4.mean():,.0f}")
print(f"True low-risk mean severity:    £{true_mean_sev4[risk4==0].mean():,.0f}")
print(f"True high-risk mean severity:   £{true_mean_sev4[risk4==1].mean():,.0f}")
```

### Task 1: Fit the severity model on claims-only

Fit a CatBoost Gamma model on the claims-only subset. Predict severity for the claims-only test set.

```python
# Train/test split: temporal (use risk as year proxy for simplicity)
train_mask4 = rng4.choice([True, False], n4, p=[0.75, 0.25])
df4_train   = df4[train_mask4]
df4_test    = df4[~train_mask4]

# Claims-only training set
df4_train_claims = df4_train[df4_train["claim_count"] > 0].copy()
df4_train_claims["mean_sev"] = (
    df4_train_claims["incurred_loss"] / df4_train_claims["claim_count"]
)

X_sev_tr4 = df4_train_claims[FEATURES4]
y_sev_tr4 = df4_train_claims["mean_sev"].values

sev_pool4 = Pool(X_sev_tr4, y_sev_tr4)
sev_model4 = CatBoostRegressor(
    loss_function="Tweedie:variance_power=2",
    iterations=200, depth=4, random_seed=42, verbose=0
)
sev_model4.fit(sev_pool4)
print("Severity model fitted.")
```

### Task 2: Three approaches to severity imputation

For the test set, compute the pure premium three ways:

**Approach A (the bug):** Predict severity for claims-only policies. Fill zero-claim policies with `sev_pred_claims_only.mean()`.

**Approach B (the fix):** Predict severity for ALL test policies (not just those with claims).

**Approach C (the oracle):** Use the true severity values from the data generating process.

```python
df4_test_claims = df4_test[df4_test["claim_count"] > 0].copy()
df4_test_claims["mean_sev"] = (
    df4_test_claims["incurred_loss"] / df4_test_claims["claim_count"]
)

# Approach A: claims-only prediction, fill others with claims-only mean
pred_claims_only4 = sev_model4.predict(Pool(df4_test_claims[FEATURES4]))
claims_only_mean4 = pred_claims_only4.mean()

# Build full test predictions (imputed)
sev_pred_A = np.where(
    df4_test["claim_count"].values > 0,
    sev_model4.predict(Pool(df4_test[FEATURES4])),  # will be same as pred_claims_only4 for those with claims
    claims_only_mean4  # impute with claims-only mean -- this is the bug
)

# Approach B: predict for all policies
sev_pred_B = sev_model4.predict(Pool(df4_test[FEATURES4]))

# Approach C: oracle
sev_oracle4 = true_mean_sev4[~train_mask4]

# Pure premium = frequency prediction * severity prediction
# For simplicity, use the true frequency here to isolate the severity effect
freq_pred4 = true_freq4[~train_mask4]

pp_A = freq_pred4 * sev_pred_A
pp_B = freq_pred4 * sev_pred_B
pp_C = freq_pred4 * sev_oracle4

# Compare by risk group
risk_test4 = risk4[~train_mask4]
print("Mean pure premium by risk group:")
print(f"{'Group':<12} {'Approach A (bug)':>18} {'Approach B (fix)':>18} {'Oracle':>12}")
for r, label in [(0, "Low risk"), (1, "High risk")]:
    mask_r = risk_test4 == r
    print(f"{label:<12} "
          f"£{pp_A[mask_r].mean():>16,.2f} "
          f"£{pp_B[mask_r].mean():>16,.2f} "
          f"£{pp_C[mask_r].mean():>10,.2f}")
```

### Task 3: Quantify the bias

For the low-risk group (most likely to have zero claims), Approach A should produce a higher mean pure premium than Approach B. This is the upward bias from the claims-only imputation.

- By what percentage does Approach A overstate the pure premium for low-risk policies?
- By what percentage does it understate for high-risk policies?
- How would this bias affect the rate optimiser's output?

### Task 4: Portfolio-level loss ratio impact

Compute the model-estimated loss ratio under Approaches A and B. Use the test set's actual incurred losses as the numerator. If the book is currently charged at 1.2x the technical premium, how does the difference between A and B affect the estimated LR?

---

## Exercise 5: Conformal calibration -- temporal vs random split

**References:** Tutorial Part 12.

**What this exercise covers:** Why the conformal calibration split must be temporal, and how to measure the impact of using a random split.

**Setup.**

```python
from insurance_conformal import InsuranceConformalPredictor
import polars as pl
import numpy as np
from catboost import CatBoostRegressor, Pool

rng5 = np.random.default_rng(seed=555)
n5   = 40_000

# Simulate severity with a time trend: claims inflate 5% per year
accident_year5 = rng5.choice([2021,2022,2023,2024], n5)
inflation_mult = (1.05 ** (accident_year5 - 2021))

vehicle_group5 = rng5.integers(1, 51, n5)
exposure5      = np.clip(rng5.beta(8, 2, n5), 0.1, 1.0)
freq5          = 0.07 * np.exp(0.01 * vehicle_group5)
claims5        = rng5.poisson(freq5 * exposure5)

true_mean_sev5 = 2_500 * inflation_mult * (1 + 0.008 * vehicle_group5)
severity5      = np.where(
    claims5 > 0,
    rng5.lognormal(np.log(true_mean_sev5), 0.8, n5),
    0.0,
)
incurred5 = np.where(claims5 > 0, severity5, 0.0)

df5 = pl.DataFrame({
    "accident_year": accident_year5.tolist(),
    "vehicle_group": vehicle_group5.tolist(),
    "exposure":      exposure5.tolist(),
    "claim_count":   claims5.tolist(),
    "incurred_loss": incurred5.tolist(),
})

print("Observed mean severity by accident year (inflating):")
df5_claims = df5.filter(pl.col("claim_count") > 0)
print(df5_claims.group_by("accident_year").agg(
    pl.col("incurred_loss").mean().round(0).alias("mean_sev")
).sort("accident_year"))
```

### Task 1: Fit the severity model

Train a CatBoost Gamma model on accident years 2021-2022 (claims only). This represents the model from the previous rate review.

```python
FEATURES5 = ["vehicle_group"]

df5_pd = df5.to_pandas()
df5_train = df5_pd[(df5_pd["accident_year"] <= 2022) & (df5_pd["claim_count"] > 0)].copy()
df5_test  = df5_pd[(df5_pd["accident_year"] == 2024) & (df5_pd["claim_count"] > 0)].copy()

df5_train["mean_sev"] = df5_train["incurred_loss"] / df5_train["claim_count"]
df5_test["mean_sev"]  = df5_test["incurred_loss"]  / df5_test["claim_count"]

sev_pool5 = Pool(df5_train[FEATURES5], df5_train["mean_sev"].values)
sev_model5 = CatBoostRegressor(
    loss_function="Tweedie:variance_power=2",
    iterations=200, depth=4, random_seed=42, verbose=0
)
sev_model5.fit(sev_pool5)
print("Severity model fitted on 2021-2022 claims.")
```

### Task 2: Random vs temporal calibration split

Calibrate two conformal predictors:

**Predictor A (wrong):** Random 80/20 split of all available data after 2022 for calibration.

**Predictor B (correct):** Temporal split -- use 2023 claims for calibration, 2024 for test.

```python
df5_post_train = df5_pd[(df5_pd["accident_year"] > 2022) & (df5_pd["claim_count"] > 0)].copy()
df5_post_train["mean_sev"] = (
    df5_post_train["incurred_loss"] / df5_post_train["claim_count"]
)

# Predictor A: random calibration split
from sklearn.model_selection import train_test_split
cal_rand, test_rand = train_test_split(df5_post_train, test_size=0.5, random_state=42)

cp_A = InsuranceConformalPredictor(
    model=sev_model5, nonconformity="pearson_weighted",
    distribution="tweedie", tweedie_power=2.0,
)
cp_A.calibrate(cal_rand[FEATURES5], cal_rand["mean_sev"].values)

# Predictor B: temporal calibration split
df5_cal_b  = df5_pd[(df5_pd["accident_year"] == 2023) & (df5_pd["claim_count"] > 0)].copy()
df5_test_b = df5_pd[(df5_pd["accident_year"] == 2024) & (df5_pd["claim_count"] > 0)].copy()
df5_cal_b["mean_sev"]  = df5_cal_b["incurred_loss"]  / df5_cal_b["claim_count"]
df5_test_b["mean_sev"] = df5_test_b["incurred_loss"] / df5_test_b["claim_count"]

cp_B = InsuranceConformalPredictor(
    model=sev_model5, nonconformity="pearson_weighted",
    distribution="tweedie", tweedie_power=2.0,
)
cp_B.calibrate(df5_cal_b[FEATURES5], df5_cal_b["mean_sev"].values)
```

### Task 3: Compare coverage diagnostics

For both predictors, validate coverage on the 2024 test set. What does the coverage-by-decile diagnostic show for each? Which predictor is better calibrated?

```python
X_te5 = df5_test_b[FEATURES5]
y_te5 = df5_test_b["mean_sev"].values

diag_A = cp_A.coverage_by_decile(X_te5, y_te5, alpha=0.10)
diag_B = cp_B.coverage_by_decile(X_te5, y_te5, alpha=0.10)

print("Coverage by decile -- Predictor A (random split):")
print(diag_A[["decile", "coverage"]])

print("\nCoverage by decile -- Predictor B (temporal split):")
print(diag_B[["decile", "coverage"]])

min_cov_A = float(diag_A["coverage"].min())
min_cov_B = float(diag_B["coverage"].min())

print(f"\nMin decile coverage -- A: {min_cov_A:.3f}  B: {min_cov_B:.3f}")
print("Conclusion:")
if min_cov_B > min_cov_A:
    print("  Temporal split (B) achieves better coverage on the 2024 test set.")
    print("  The random split (A) uses some 2024 data in calibration, artificially")
    print("  improving apparent coverage. On genuinely unseen 2024 data, it fails more.")
else:
    print("  For this dataset, the difference is small -- but the principle holds.")
    print("  On a dataset with stronger time trends, the gap would be larger.")
```

### Task 4: Stretch -- what happens when the inflation rate changes?

Modify the data generating process to use a 10% per year inflation rate (instead of 5%). Re-run the calibration and coverage check. At what inflation rate does the temporal calibration split also fail to maintain 85% minimum coverage? What does this imply for recalibration frequency?

---

## Exercise 6: The audit record -- build a reproducibility test

**References:** Tutorial Parts 14 and 15.

**What this exercise covers:** Verifying that the audit record enables exact reproduction of a historical pipeline run.

### Task 1: Run the pipeline and record the audit

This task asks you to run a miniature version of the full pipeline (simplified for speed) and write an audit record.

```python
import polars as pl
import numpy as np
import pandas as pd
import json
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor, Pool
from datetime import date

# ---- Miniature pipeline ----

# Generate data
rng6 = np.random.default_rng(seed=2026)
n6   = 10_000

df6 = pl.DataFrame({
    "ncb_years":    rng6.choice([0,1,2,3,4,5], n6).tolist(),
    "exposure":     np.clip(rng6.beta(8, 2, n6), 0.05, 1.0).tolist(),
    "claim_count":  rng6.poisson(0.07, n6).tolist(),
    "accident_year": rng6.choice([2021,2022,2023,2024], n6).tolist(),
})

FEATURES6 = ["ncb_years"]
CAT6      = []

# Write to Delta
spark.createDataFrame(df6.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema","true") \
    .saveAsTable("main.motor_q2_2026.mini_raw")

v_raw6 = spark.sql(
    "DESCRIBE HISTORY main.motor_q2_2026.mini_raw LIMIT 1"
).collect()[0]["version"]

# Split and train
df6_pd   = df6.to_pandas()
train6   = df6_pd[df6_pd["accident_year"] < 2024]
test6    = df6_pd[df6_pd["accident_year"] == 2024]

pool_tr6 = Pool(train6[FEATURES6], train6["claim_count"].values,
                baseline=np.log(np.clip(train6["exposure"].values, 1e-6, None)))
pool_te6 = Pool(test6[FEATURES6], test6["claim_count"].values,
                baseline=np.log(np.clip(test6["exposure"].values, 1e-6, None)))

with mlflow.start_run(run_name="mini_pipeline_ex6") as run6:
    model6 = CatBoostRegressor(
        loss_function="Poisson", iterations=150, depth=4, random_seed=42, verbose=0
    )
    model6.fit(pool_tr6, eval_set=pool_te6)
    pred6  = model6.predict(pool_te6)
    dev6   = poisson_deviance(test6["claim_count"].values, pred6,
                               test6["exposure"].values)
    mlflow.log_metric("test_deviance", dev6)
    mlflow.log_param("raw_table_version", int(v_raw6))
    mlflow.catboost.log_model(model6, "model6")
    run_id6 = run6.info.run_id

audit6 = {
    "run_date":           str(date.today()),
    "raw_table":          "main.motor_q2_2026.mini_raw",
    "raw_table_version":  int(v_raw6),
    "model_run_id":       run_id6,
    "n_train":            len(train6),
    "n_test":             len(test6),
    "test_deviance":      round(dev6, 5),
    "features":           json.dumps(FEATURES6),
}
print("Audit record:")
print(json.dumps(audit6, indent=2))
```

### Task 2: Simulate a data correction

Add 500 synthetic corrections to the raw data -- modify the claim counts for 500 randomly selected policies.

```python
rng6b = np.random.default_rng(seed=9999)
correction_idx = rng6b.choice(n6, 500, replace=False)

df6_corrected = df6.to_pandas()
df6_corrected.loc[correction_idx, "claim_count"] = (
    df6_corrected.loc[correction_idx, "claim_count"] + rng6b.integers(0, 3, 500)
).clip(lower=0)

spark.createDataFrame(df6_corrected) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema","true") \
    .saveAsTable("main.motor_q2_2026.mini_raw")

v_raw6_after = spark.sql(
    "DESCRIBE HISTORY main.motor_q2_2026.mini_raw LIMIT 1"
).collect()[0]["version"]

print(f"Original version: {v_raw6}")
print(f"After correction: {v_raw6_after}")
```

### Task 3: Reproduce the original pipeline output

Using only the audit record from Task 1, reproduce the original model output. Verify that the test deviance matches the logged deviance exactly.

```python
# Step 1: Read the data at the logged version
original_data = spark.read.format("delta") \
    .option("versionAsOf", audit6["raw_table_version"]) \
    .table(audit6["raw_table"]) \
    .toPandas()

print(f"Reproduced data rows: {len(original_data):,}")
print(f"Original n_train + n_test: {audit6['n_train'] + audit6['n_test']:,}")

# Step 2: Load the original model from MLflow
original_model = mlflow.catboost.load_model(
    f"runs:/{audit6['model_run_id']}/model6"
)

# Step 3: Reproduce the test set and score it
orig_train = original_data[original_data["accident_year"] < 2024]
orig_test  = original_data[original_data["accident_year"] == 2024]

orig_pool_te = Pool(orig_test[FEATURES6], orig_test["claim_count"].values,
                    baseline=np.log(np.clip(orig_test["exposure"].values, 1e-6, None)))

repro_pred = original_model.predict(orig_pool_te)
repro_dev  = poisson_deviance(
    orig_test["claim_count"].values, repro_pred, orig_test["exposure"].values
)

print(f"\nLogged deviance:     {audit6['test_deviance']:.5f}")
print(f"Reproduced deviance: {repro_dev:.5f}")
print(f"Match: {abs(repro_dev - audit6['test_deviance']) < 1e-8}")
```

**Expected outcome:** The reproduced deviance should match the logged deviance to floating-point precision. If it does not, identify why -- the most common causes are: the model was not logged with the exact same random seed, or the test set rows changed between versions.

### Task 4: Show the current data gives a different result

Prove that using the current (corrected) data instead of the original gives a different deviance, even with the same model:

```python
# Score the original model on the CURRENT (corrected) data
current_test = df6_corrected[df6_corrected["accident_year"] == 2024]
current_pool = Pool(current_test[FEATURES6], current_test["claim_count"].values,
                    baseline=np.log(np.clip(current_test["exposure"].values, 1e-6, None)))

current_pred = original_model.predict(current_pool)
current_dev  = poisson_deviance(
    current_test["claim_count"].values, current_pred, current_test["exposure"].values
)

print(f"Original data deviance:  {repro_dev:.5f}")
print(f"Corrected data deviance: {current_dev:.5f}")
print(f"Difference:              {abs(current_dev - repro_dev):.5f}")
print("\nThis demonstrates why the audit record's raw_table_version is essential:")
print("without it, you cannot distinguish between the original result and")
print("a result contaminated by post-hoc data corrections.")
```

---

## Exercise 7: Putting it all together -- full rate review in one session

**References:** All tutorial parts.

**What this exercise covers:** Running a complete rate review cycle on your own synthetic data from scratch, without using the tutorial's configuration. This is the capstone exercise.

**Objective.** You are the lead pricing actuary for the Q3 2026 UK motor rate review. You need to produce a pricing committee pack by the end of the session. The output must include: a Poisson frequency model validated on accident year 2024, conformal 90% severity intervals validated on accident year 2024, a rate action recommendation, and a full pipeline audit record.

**Requirements:**
1. Generate a portfolio of 100,000 policies with at least four rating factors of your own design. At least one must be a genuine categorical variable. At least one must require a non-trivial encoding transform.
2. Use a TRANSFORMS list pattern as in Tutorial Part 7. Define all transforms as pure functions.
3. Run three-fold walk-forward CV with the Poisson deviance.
4. Run Optuna tuning (minimum 15 trials) for both frequency and severity.
5. Calibrate a conformal predictor on the penultimate accident year. Validate on the final year. Achieve at least 85% minimum decile coverage or explain why you cannot.
6. Write a pipeline audit record to a Delta table.
7. Write a five-bullet pricing committee summary (as a Markdown cell) covering: model performance (Gini and deviance), conformal coverage result, pure premium range, recommended rate action and rationale, limitations.

**Time allocation:** This exercise is designed to take 45-60 minutes. Do not try to write perfect code on the first pass -- get the pipeline running, then refine.

### Starter code

```python
# Stage 0: Libraries (run this cell first, then restart Python)
%pip install catboost optuna mlflow polars "insurance-cv" "insurance-conformal[catboost]" "rate-optimiser" --quiet
```

```python
dbutils.library.restartPython()
```

```python
# Your own portfolio -- customise the features and DGP
import polars as pl
import numpy as np
import optuna
import mlflow
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor

# YOUR PORTFOLIO DGP HERE
# Requirements:
# - 100,000 policies
# - 4+ rating factors, at least 1 categorical, at least 1 needing a non-trivial transform
# - accident years 2021-2024
# - claim_count drawn from Poisson(frequency * exposure)
# - incurred_loss representing severity * claim_count

rng = np.random.default_rng(seed=2026)
n = 100_000

# ... your code here ...
```

```python
# YOUR TRANSFORMS LIST HERE
# Requirements:
# - At least 4 transform functions, one per rating factor
# - All constants defined at module level (not inside functions)
# - apply_transforms() calls all functions in order

TRANSFORMS   = []  # fill in
FEATURE_COLS = []  # fill in
CAT_FEATURES = []  # fill in

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    for fn in TRANSFORMS:
        df = fn(df)
    return df
```

```python
# YOUR CV LOOP HERE
# Requirements:
# - 3 temporal folds
# - Poisson deviance metric
# - IBNR buffer of 6 months (or justify a different buffer for your DGP)
# - baseline=np.log(exposure) for the Poisson offset -- not weight=

cv_deviances = []
# ... your code here ...
```

```python
# YOUR OPTUNA STUDIES HERE
# Requirements:
# - Separate studies for frequency and severity
# - Minimum 15 trials each
# - Tune on the last CV fold
# - best_freq_params and best_sev_params must be defined before Stage 6

best_freq_params = {}
best_sev_params  = {}
# ... your code here ...
```

```python
# YOUR FINAL MODELS HERE
# Requirements:
# - Log to MLflow with data table versions and feature list
# - Predict severity for ALL test policies (not just claims-only)
# - Compute pure_premium = freq_rate * sev_pred for all test policies

freq_run_id = ""
sev_run_id  = ""
pure_premium = np.array([])
# ... your code here ...
```

```python
# YOUR CONFORMAL INTERVALS HERE
# Requirements:
# - Temporal calibration: calibration year = penultimate accident year
# - MUST NOT use random split for calibration
# - Run coverage_by_decile diagnostic
# - Flag any risks in top 10% relative interval width for referral

min_cov = 0.0
# ... your code here ...
```

```python
# YOUR AUDIT RECORD HERE
# Requirements:
# - Written to a Delta table in mode("append")
# - Must include: raw table version, features table version, MLflow run IDs,
#   test deviance, conformal min decile coverage, run date, feature list

# ... your code here ...
```

```python
%md
## Q3 2026 Motor Rate Review -- Pricing Committee Summary

- **Model performance:** (your Gini and test deviance here)
- **Conformal coverage:** (your coverage diagnostic result here)
- **Pure premium range:** (your P25-P75 and P95 here)
- **Recommended rate action:** (your recommendation here)
- **Key limitations:** (at least two specific to your portfolio)
```

### Stretch tasks

**Stretch 1.** Add the `FeatureSpec` class from Exercise 1. Record the spec at training time, log it as an MLflow artefact, and write a validation check that runs before scoring.

**Stretch 2.** Add a fourth accident year to your portfolio (2025) with a 6% loss cost trend uplift. Retrain the pipeline on 2021-2024 and predict frequencies for 2025. Use these predictions as the technical premium input for the rate optimiser. Set the LR target at 2% below the observed test-set LR. Report the factor adjustments.

**Stretch 3.** The efficient frontier. Solve the rate optimiser for LR targets of 0.68, 0.70, 0.72, 0.74, and 0.76. Plot the resulting volume-LR frontier (volume retention on x, expected LR on y). At which LR target does the volume constraint bind first?

**Stretch 4.** Audit trail integrity check. After completing the full pipeline, simulate a data correction (modify 200 rows of the raw table) and recheck the audit record. Show that: (a) the Delta version increments, (b) the original data is still accessible at the original version, (c) the logged model's deviance is unchanged when scored against the original data, and (d) the deviance changes when scored against the corrected data.

---

## Appendix: Common error messages and fixes

### `ModuleNotFoundError: No module named 'insurance_cv'`

The `%pip install insurance-cv` cell did not run, or ran before `dbutils.library.restartPython()`. Run the install cell again, wait for it to complete, then run `dbutils.library.restartPython()` before importing.

### `CatBoostError: Cannot find model for feature ...`

CatBoost is seeing a feature in scoring that was treated as categorical in training (or vice versa). The `cat_features` list at scoring time must be identical to the one used in training. Use the `FeatureSpec` class from Exercise 1 to detect this before scoring.

### `ValueError: baseline and label must have the same number of elements`

The `baseline` array passed to `Pool()` has a different length to the label array. Check that `np.log(exposure)` is computed from the same subset as `y_train` and `X_train`.

### `AssertionError` in audit record cell

One of the pipeline stages did not set its output variable. Check that `freq_run_id`, `sev_run_id`, `min_cov`, and `result.converged` are all defined before the audit record cell runs. If the pipeline failed partway through, re-run from the failed stage.

### `AnalysisException: Table or view not found`

The schema does not exist. Run the `CREATE SCHEMA` cell in Stage 1 before writing any tables. If using `hive_metastore`, the schema creation syntax differs -- see Tutorial Part 5.

### `optuna.exceptions.ExperimentalWarning`

Informational only -- not an error. Suppress with `optuna.logging.set_verbosity(optuna.logging.WARNING)` at the top of the Optuna cell.

### Coverage diagnostic fails with `IndexError`

The calibration set has too few observations (fewer than 100 claims) to split into deciles reliably. Either extend the calibration window (use two years instead of one) or reduce the number of coverage groups with `coverage_by_quantile(n_groups=5)`.
