# Module 9 Exercises: Demand Modelling and Price Elasticity

Ten exercises. Work through them in order - each builds on the data and models from previous ones. Solutions are in collapsed sections at the end of each exercise.

Before starting: read Parts 1-17 of the tutorial. All concepts used here are explained there.

The same install cell from the tutorial applies. If you are continuing in the same notebook session, the libraries are already installed and you can skip the `%pip install` cell.

**Note on datasets:** The conversion and renewal datasets used in this module (`df_quotes` from `generate_conversion_data` and `df_renewals` from `make_renewal_data`) are separate from the motor portfolio used in Modules 1--8. They come from different data generators designed specifically for demand modelling. The setup block below regenerates them from scratch.

**Note on session state:** Several exercises reference objects (`est_renewal`, `est_forest`, etc.) created in earlier exercises or in the tutorial. If you are starting partway through, re-run the relevant earlier cells first before attempting an exercise.

---

## Setup: generate the base datasets

Run this in a new cell before starting Exercise 1. All exercises use these datasets.

```python
import numpy as np
import polars as pl
from scipy.special import expit

from insurance_demand import ConversionModel, RetentionModel, ElasticityEstimator
from insurance_demand.datasets import generate_conversion_data, generate_retention_data
from insurance_demand.compliance import ENBPChecker

from insurance_elasticity.data import make_renewal_data
from insurance_elasticity.fit import RenewalElasticityEstimator
from insurance_elasticity.diagnostics import ElasticityDiagnostics
from insurance_elasticity.optimise import RenewalPricingOptimiser
from insurance_elasticity.demand import demand_curve

# Conversion dataset: 150,000 quotes, true elasticity = -2.0
df_quotes = generate_conversion_data(n_quotes=150_000, seed=42)

# Renewal dataset: 50,000 policies, heterogeneous true elasticity
df_renewals = make_renewal_data(n=50_000, seed=42)

# Add lapsed column for retention model
df_renewals = df_renewals.with_columns(
    (1 - pl.col("renewed")).alias("lapsed")
)

print(f"Quotes:   {len(df_quotes):,} rows, {df_quotes['converted'].mean():.1%} conversion rate")
print(f"Renewals: {len(df_renewals):,} rows, {df_renewals['renewed'].mean():.1%} renewal rate")
```

---

## Exercise 1: Building a basic conversion model

**Reference:** Tutorial Parts 3-5

**Estimated time:** 20 minutes

You are the pricing analyst at a UK motor insurer. Your team has collected 150,000 new business quotes from the last year across four PCW channels and a direct channel. Your task is to build the first conversion model the team has had, validate it, and present the key findings.

### Part A: Fit the logistic conversion model

Fit a `ConversionModel` with `base_estimator="logistic"` using the following features: `age`, `vehicle_group`, `ncd_years`, `area`, and `channel`. Include `rank_position_col="rank_position"`.

Print the coefficient table from `conv_model.summary()`. Answer these questions by inspecting the table:

1. Is the coefficient on `log_price_ratio` negative? Should it be, and why?
2. Which non-price feature has the largest absolute coefficient?
3. What does an odds ratio of 0.85 on a feature mean in plain English?

### Part B: One-way validation

Run `conv_model.oneway(df_quotes, "channel")`. For each channel, report the observed conversion rate, the fitted conversion rate, and the lift.

If any channel has lift above 1.25 or below 0.80, describe what that tells you about the model's performance for that channel and what you would do to fix it.

### Part C: Interpret the rank position effect

Run:
```python
# Create a small test dataset with varying rank position
base_row = df_quotes.head(1)
results = []
for rank in range(1, 7):
    row = base_row.with_columns([
        pl.lit(rank).alias("rank_position"),
        pl.lit("pcw_confused").alias("channel"),
    ])
    results.append({"rank_position": rank,
                    "conv_prob": float(conv_model.predict_proba(row).to_numpy()[0])})
pl.DataFrame(results)
```

Look at how conversion probability changes as rank position goes from 1 (cheapest) to 6. Is the relationship linear? Why does a PCW rank position of 1 give such a large advantage compared to rank 2?

<details>
<summary>Solution</summary>

### Part A

```python
conv_model = ConversionModel(
    base_estimator="logistic",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
    logistic_C=1.0,
)
conv_model.fit(df_quotes)
print(conv_model.summary())
```

The coefficient on `log_price_ratio` should be negative (around -1.5 to -2.5). It is negative because a higher price relative to the technical premium reduces conversion probability. Higher price, lower probability of buying: this is the demand law.

The non-price feature with the largest absolute coefficient is typically `log_rank` (the log-transformed rank position) or a channel dummy. Being first on the PCW versus second has a large effect on conversion that is not fully captured by the price ratio alone - customers have a bias toward the top result.

An odds ratio of 0.85 means: the odds of converting are 15% lower for a one-unit increase in that feature. For a binary feature like being in a specific channel, it means that channel has 15% lower odds of conversion than the reference category.

### Part B

```python
channel_ow = conv_model.oneway(df_quotes, "channel")
print(channel_ow.to_string())
```

For this synthetic dataset, lift should be close to 1.0 for all channels if the model is well-specified. If a channel shows lift above 1.25, it means the model is under-predicting conversion for that channel. Remedies: add channel interaction terms, use a separate model per channel, or upgrade to CatBoost which captures non-linear interactions automatically.

For PCW channels, you would typically expect higher conversion rates than direct at the same price ratio - these customers have already made their shortlist and are comparing your quote to a small set of alternatives. The model should capture this via the channel dummies.

### Part C

The relationship is not linear because the feature is `log(rank_position)`, not `rank_position` directly. This is intentional: the step from rank 1 to rank 2 is much larger than from rank 4 to rank 5. Being cheapest on a PCW puts you in the default "recommended" sort at the top of the page. Being second cheapest means most customers have to scroll to see you. The log transformation captures this diminishing marginal cost of moving down the rankings.

On a PCW with typically 8-15 quotes displayed, the top 3 positions attract most of the clicks. The conversion probability cliff between rank 1 and rank 2 is often 40-60% relative in observational data.

</details>

---

## Exercise 2: Diagnosing confounding

**Reference:** Tutorial Part 3

**Estimated time:** 25 minutes

This exercise makes the confounding problem visible. You will run both a naive logistic regression and the DML estimator, compare their outputs, and explain the difference.

### Part A: The naive estimate

Fit a simple logistic regression of `converted` on `log_price_ratio` only (no other features). This is the most naive possible model - it just regresses conversion on price with no controls.

```python
naive_model = ConversionModel(
    base_estimator="logistic",
    feature_cols=[],  # no features - only price
    rank_position_col=None,
)
naive_model.fit(df_quotes)
naive_summary = naive_model.summary()
print("Naive model (no controls):")
print(naive_summary)
```

Record the coefficient on `log_price_ratio`. Call it `beta_naive`.

### Part B: The controlled logistic estimate

Now fit the same model but with all the risk features as controls:

```python
controlled_model = ConversionModel(
    base_estimator="logistic",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
)
controlled_model.fit(df_quotes)
controlled_summary = controlled_model.summary()
```

Record the coefficient on `log_price_ratio`. Call it `beta_controlled`.

### Part C: The DML estimate

Fit the DML estimator and record the global ATE. This should already be done from the tutorial. If you have the `est_conversion` object from Part 8 of the tutorial, use it. Otherwise:

```python
est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
)
est.fit(df_quotes)
print(est.summary())
```

### Part D: Compare and explain

Fill in this table:

| Estimator | Coefficient | Bias vs. true (-2.0) |
|-----------|-------------|----------------------|
| Naive (no controls) | | |
| Logistic with controls | | |
| DML | | |
| True elasticity | -2.0 | - |

Write 3-4 sentences explaining why the naive estimate is more biased than the controlled logistic, and why the DML estimate is more accurate than either.

<details>
<summary>Solution</summary>

### Part D: Explanation

The naive model regresses conversion on price with no other variables. In the data generating process, high-risk customers (young age, high vehicle group) receive both higher prices (because their technical premium is higher) and lower conversion rates (because they have fewer market alternatives at any given price). The naive regression sees "higher prices, lower conversion" and attributes this to price sensitivity. But some of the lower conversion is due to the high-risk customers having fewer options, not just the higher price. The naive coefficient is too negative.

The controlled logistic model includes risk features, which absorbs most of the between-segment variation. The coefficient is closer to the truth. But it is still biased because the logistic model does not correctly separate the within-segment price effect from the residual between-segment risk variation. The OLS / logistic regression of Y on D, X is consistent only if the functional form is correctly specified and there is no omitted variable. Both conditions are violated here.

The DML estimator partialls out the confounders from both Y and D before estimating the price coefficient. What remains in D_tilde is the variation in `log_price_ratio` that is not explained by the risk features. In this dataset, that variation comes from the quarterly rate review loading and random quote-level commercial decisions. It is approximately exogenous. Regressing Y_tilde on D_tilde recovers the pure causal price effect, which is close to the true -2.0.

The practical lesson: use DML for any elasticity estimate used in pricing decisions. The naive and controlled logistic approaches are fine for volume forecasting at current prices but wrong for elasticity.

</details>

---

## Exercise 3: Retention model and price sensitivity

**Reference:** Tutorial Parts 5-6

**Estimated time:** 25 minutes

Your renewal book has 50,000 policies. The pricing director asks: "how much do our customers care about price increases, and does it vary by tenure?"

### Part A: Fit the retention model

Fit a logistic retention model on `df_renewals` (with the `lapsed` column already added in the setup). Use features: `tenure_years`, `ncd_years`, `payment_method`, `age`, `channel`.

Include `price_change_col="log_price_change"` and set `cat_features=["payment_method", "channel"]`.

### Part B: Price sensitivity by tenure band

Create a tenure band variable:

```python
df_renewals = df_renewals.with_columns(
    pl.when(pl.col("tenure_years") < 2).then(pl.lit("0-1yr"))
    .when(pl.col("tenure_years") < 5).then(pl.lit("2-4yr"))
    .when(pl.col("tenure_years") < 8).then(pl.lit("5-7yr"))
    .otherwise(pl.lit("8yr+"))
    .alias("tenure_band")
)
```

Now compute the price sensitivity (dP(lapse)/d(log_price_change)) for each policy and report the mean by tenure band. Which tenure band is most price-sensitive?

### Part C: The renewal paradox

You find that long-tenure customers are less price-sensitive than short-tenure customers. Your colleague suggests: "great, we can charge them more." Under PS21/5, why is this reasoning wrong? What can you legitimately do with this information?

Write a 3-4 sentence answer.

### Part D: Predicted vs. observed lapse rates

Run a one-way check of the retention model on the `lapsed` column by `ncd_years`. Report the observed and fitted lapse rates for each NCD band. At which NCD levels is the model best calibrated?

<details>
<summary>Solution</summary>

### Part A

```python
retention_model = RetentionModel(
    model_type="logistic",
    outcome_col="lapsed",
    price_change_col="log_price_change",
    feature_cols=["tenure_years", "ncd_years", "payment_method", "age", "channel"],
    cat_features=["payment_method", "channel"],
)
retention_model.fit(df_renewals)
```

### Part B

```python
sensitivity = retention_model.price_sensitivity(df_renewals)

sensitivity_pl = (
    df_renewals
    .select("tenure_band")
    .with_columns(pl.Series("sensitivity", sensitivity.to_numpy()))
    .group_by("tenure_band")
    .agg([
        pl.col("sensitivity").mean().alias("mean"),
        pl.col("sensitivity").std().alias("std"),
        pl.len().alias("count"),
    ])
    .sort("tenure_band")
)
print(sensitivity_pl)
```

Short-tenure customers (0-1yr) should be the most price-sensitive. They are less embedded in the insurer relationship and have less invested in the tenure - they do not have 5 years of NCD protection to think about. Long-tenure customers (8yr+) have higher inertia, built up through payment habit, NCD protection concern, and the effort of switching.

### Part C

Under PS21/5, a firm cannot use lapse propensity (i.e., inertia) to set a higher renewal price. Charging a long-tenure customer more because you know they are unlikely to leave is exactly the loyalty penalty the rule bans. You cannot price towards the ENBP ceiling only for inelastic customers while discounting elastic customers. The ENBP ceiling applies regardless of estimated lapse probability.

What you can legitimately do: identify short-tenure, high-price-sensitivity customers and offer them a targeted retention discount (pricing below ENBP is permitted). The discount is moving in the direction of lower price, not higher. This is the acceptable commercial use of the lapse model.

### Part D

```python
ncd_ow = retention_model.oneway(df_renewals, "ncd_years")
print(ncd_ow)
```

The model is typically best calibrated at mid-NCD levels (NCD 2-4 years) where sample sizes are largest. At NCD=0 and NCD=5 (the extremes), there may be more lift deviation because these groups have distinct behavioural characteristics - NCD=0 customers are often first-time renewing young drivers; NCD=5 customers are the most tenured and experienced. If lift is above 1.2 at these extremes, add an interaction between `ncd_years` and `tenure_years`, or consider a CatBoost backend.

</details>

---

## Exercise 4: The near-deterministic price problem in practice

**Reference:** Tutorial Part 7

**Estimated time:** 20 minutes

This exercise simulates a real-world data quality problem and forces you to interpret the diagnostic output correctly.

### Part A: Generate a near-deterministic dataset

```python
df_ndp = make_renewal_data(n=50_000, seed=42, near_deterministic=True)
```

This dataset was generated with `price_variation_sd=0.01` - almost all the price change is determined by the re-rating formula, leaving very little exogenous variation.

Run the treatment variation diagnostic on this dataset:

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
diag = ElasticityDiagnostics()
report_ndp = diag.treatment_variation_report(
    df_ndp,
    treatment="log_price_change",
    confounders=confounders,
)
print(report_ndp.summary())
```

### Part B: Interpret the diagnostic output

The report contains several statistics. For each, write one sentence explaining what it tells you:

1. `Var(D)` - the total variance of the price change
2. `Var(D_tilde)` - the residual variance after conditioning on confounders
3. `Var(D_tilde)/Var(D)` - the variation fraction
4. `Treatment nuisance R²` - the R-squared of the treatment nuisance model

### Part C: Consequences of ignoring the warning

Suppose you ignored the `weak_treatment` warning and fitted the DML model anyway. What would happen to:

1. The point estimate (would it be biased up, down, or randomly?)
2. The confidence interval (wider or narrower than on good data?)
3. The practical usefulness of the output for pricing decisions

Write a short paragraph.

### Part D: Designing a quasi-experiment (advanced)

**Advanced.** This section implements an instrumental variables approach via PLIV (partially linear IV regression), which goes beyond the tutorial content. It is intended as a stretch task.

You are the pricing actuary. The diagnostic has told you that your data does not have sufficient treatment variation. You cannot run a randomised A/B price test without board approval (which will take 3 months). The report suggests using "bulk re-rate quasi-experiments."

Your insurer applied a uniform 8% rate increase to all motor renewals in Q1 2024 (affecting all customers with January to March anniversary dates). Customers outside this quarter saw different rate changes based on market conditions.

Write the Polars code to create an indicator variable for this quasi-experiment, and explain in 2-3 sentences why this variable would improve the DML identification.

<details>
<summary>Solution</summary>

### Part B

1. `Var(D)` - the total variance of log_price_change across the portfolio. A very small value means almost all customers received nearly identical price changes.
2. `Var(D_tilde)` - the variance of the part of price change that is not explained by the observable risk features. This is what DML uses to identify the causal effect. Near zero = nothing to work with.
3. `Var(D_tilde)/Var(D)` - the fraction of price variation that is exogenous (not predicted by confounders). Below 0.10 means DML is analogous to an IV estimator with a very weak instrument.
4. `Treatment nuisance R²` - how well the observable risk features predict the price change. An R-squared above 0.90 means the pricing system is nearly deterministic: knowing the risk features tells you almost exactly what price will be offered.

### Part C

If you ignore the warning and fit DML anyway: the point estimate will have high variance (because D_tilde has near-zero variance, the regression in Step 3 is numerically unstable - small changes in the nuisance model estimation lead to large swings in the coefficient). The confidence interval will be very wide - sometimes you will see CIs spanning [-10, +5], which is useless for pricing. The practical consequence is that the output cannot be used to make pricing decisions. You might present a point estimate of -2.0 with a CI of [-8.5, +4.5], which contains zero and spans the range from "catastrophically elastic" to "mildly anti-elastic." This is not a usable input to a pricing optimiser.

### Part D

```python
# Assume renewal_date column exists in the format yyyy-mm-dd
df_ndp = df_ndp.with_columns([
    pl.col("renewal_date").dt.month().alias("renewal_month"),
]).with_columns(
    pl.when(pl.col("renewal_month").is_in([1, 2, 3]))
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .alias("q1_bulk_rerate_indicator")
)
```

The Q1 bulk re-rate indicator is exogenous at the individual customer level because the timing of a customer's anniversary date is determined by when they originally bought their policy, not by their current risk profile or price sensitivity. Customers with January-March anniversary dates received the 8% increase simply because they renewed in Q1; customers with April-June anniversary dates did not. This creates cross-sectional variation in price change that is independent of the observable confounders, which is exactly the variation DML needs. You would pass this as `instrument_col` in `ElasticityEstimator` to use the PLIV (IV-DML) estimator.

</details>

---

## Exercise 5: CatBoost conversion model vs. logistic

**Reference:** Tutorial Part 5

**Estimated time:** 30 minutes

The tutorial showed that CatBoost gives higher AUC than logistic regression on the conversion data. This exercise explores when the improvement matters and when it does not.

### Part A: Fit both models

Fit a logistic and a CatBoost conversion model using the same feature set:

```python
features = ["age", "vehicle_group", "ncd_years", "area", "channel"]
cat_features = ["area", "channel"]

conv_logistic = ConversionModel(
    base_estimator="logistic",
    feature_cols=features,
    rank_position_col="rank_position",
)
conv_logistic.fit(df_quotes)

conv_catboost = ConversionModel(
    base_estimator="catboost",
    feature_cols=features,
    rank_position_col="rank_position",
    cat_features=cat_features,
)
conv_catboost.fit(df_quotes)
```

### Part B: AUC comparison

Compute the AUC for both models. Report which model performs better and by how many Gini points (Gini = 2*AUC - 1).

Note: compute AUC on the full dataset here for speed. In production you would use a held-out test set.

### Part C: One-way comparison

Run `oneway` by `ncd_years` for both models. For which NCD bands does the CatBoost model show better calibration (lift closer to 1.0)?

Explain in one paragraph why CatBoost tends to do better at the extreme NCD levels (NCD=0 and NCD=5) while both models perform similarly at mid-NCD levels.

### Part D: When does the difference matter?

The AUC improvement from CatBoost over logistic is typically 2-4 Gini points on conversion data. Complete the following analysis to determine whether this improvement translates into meaningfully different pricing decisions:

```python
# At what price ratio does each model predict 10% conversion?
# Test on a representative customer (median values across features)

# Find median feature values
median_age = int(df_quotes["age"].median())
median_vg  = int(df_quotes["vehicle_group"].median())
median_ncd = int(df_quotes["ncd_years"].median())

test_row = pl.DataFrame({
    "age": [median_age],
    "vehicle_group": [median_vg],
    "ncd_years": [median_ncd],
    "area": ["midlands"],
    "channel": ["pcw_confused"],
    "quoted_price": [500.0],
    "technical_premium": [500.0],
    "rank_position": [3],
    "converted": [0],
    "log_price_ratio": [0.0],
    "price_ratio": [1.0],
})

# Vary price ratio from 0.8 to 1.4
price_ratios = np.linspace(0.8, 1.4, 60)
results_logistic = []
results_catboost = []

for pr in price_ratios:
    row = test_row.with_columns([
        pl.lit(500.0 * pr).alias("quoted_price"),
        pl.lit(np.log(pr)).alias("log_price_ratio"),
        pl.lit(pr).alias("price_ratio"),
    ])
    results_logistic.append(float(conv_logistic.predict_proba(row).to_numpy()[0]))
    results_catboost.append(float(conv_catboost.predict_proba(row).to_numpy()[0]))
```

Plot the two demand curves side by side. At what loading does each model predict a 10% conversion rate? If the 10% conversion loading differs by more than 2%, that is a commercially significant gap. If it differs by less than 0.5%, the models are interchangeable for this segment.

<details>
<summary>Solution</summary>

### Part B

```python
from sklearn.metrics import roc_auc_score

y_true = df_quotes["converted"].to_numpy()
auc_l = roc_auc_score(y_true, conv_logistic.predict_proba(df_quotes).to_numpy())
auc_c = roc_auc_score(y_true, conv_catboost.predict_proba(df_quotes).to_numpy())

print(f"Logistic AUC: {auc_l:.4f}  Gini: {2*auc_l-1:.4f}")
print(f"CatBoost AUC: {auc_c:.4f}  Gini: {2*auc_c-1:.4f}")
print(f"Gini improvement: {(2*auc_c-1 - (2*auc_l-1)):.4f}")
```

### Part C

CatBoost does better at NCD=0 and NCD=5 because these segments have distinctive non-linear interaction patterns. NCD=0 customers are almost exclusively young first-time renewals, and the combination of young age, high vehicle group, and high price sensitivity creates interactions that the logistic model cannot capture without explicit interaction terms. NCD=5 customers are the opposite extreme: highly inelastic, tenured, and with payment patterns (mainly direct debit) that interact with their lapse behaviour in ways a simple additive logistic model misses. CatBoost's tree structure captures these automatically.

### Part D

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(price_ratios, [r * 100 for r in results_logistic], label="Logistic", linewidth=2)
ax.plot(price_ratios, [r * 100 for r in results_catboost], label="CatBoost", linewidth=2, linestyle="--")
ax.axhline(10, color="red", linewidth=0.8, linestyle=":", label="10% target")
ax.set_xlabel("Price ratio (quoted / technical)")
ax.set_ylabel("Predicted conversion rate (%)")
ax.set_title("Demand curve comparison: logistic vs. CatBoost")
ax.legend()
plt.tight_layout()
plt.show()

# Find price ratio at 10% conversion for each model
idx_l = min(range(len(results_logistic)), key=lambda i: abs(results_logistic[i] - 0.10))
idx_c = min(range(len(results_catboost)), key=lambda i: abs(results_catboost[i] - 0.10))
print(f"Logistic: 10% conversion at price ratio = {price_ratios[idx_l]:.3f}")
print(f"CatBoost: 10% conversion at price ratio = {price_ratios[idx_c]:.3f}")
```

If the price ratios at 10% conversion differ by more than 2%, the models would lead to different commercial pricing decisions for this segment. In practice, for the mid-NCD customer at a median age and area, the two models are usually within 1-2% of each other. The practical case for CatBoost is strongest for extreme segments (very young or very old, high vehicle group, London) where the non-linear interactions matter most.

</details>

---

## Exercise 6: Fitting and validating the heterogeneous elasticity model

**Reference:** Tutorial Parts 8-9

**Estimated time:** 30-40 minutes (fitting time dominated by CausalForestDML)

This exercise fits the `RenewalElasticityEstimator` and validates the output against the known ground truth in the synthetic dataset.

### Part A: Fit the CausalForestDML estimator

Fit the estimator with `cate_model="linear_dml"` first (faster) and then with `cate_model="causal_forest"`. Compare the ATE from each.

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

# LinearDML: constant treatment effect (much faster)
est_linear = RenewalElasticityEstimator(
    cate_model="linear_dml",
    catboost_iterations=300,
    n_folds=5,
)
est_linear.fit(df_renewals, outcome="renewed",
               treatment="log_price_change", confounders=confounders)

ate_linear, lb_linear, ub_linear = est_linear.ate()
print(f"LinearDML ATE: {ate_linear:.3f}  95% CI: [{lb_linear:.3f}, {ub_linear:.3f}]")
```

After LinearDML completes, fit the CausalForestDML:

```python
est_forest = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=200,
    catboost_iterations=400,
    n_folds=5,
)
est_forest.fit(df_renewals, outcome="renewed",
               treatment="log_price_change", confounders=confounders)

ate_forest, lb_forest, ub_forest = est_forest.ate()
print(f"CausalForestDML ATE: {ate_forest:.3f}  95% CI: [{lb_forest:.3f}, {ub_forest:.3f}]")
```

### Part B: GATE validation

For the `est_forest` model, compute GATEs by `ncd_years`. The true elasticities by NCD band in the synthetic DGP are:

| NCD | True elasticity |
|-----|----------------|
| 0   | -3.5 * 0.6 + -2.5 * 0.4 = -3.1 (approx, age-weighted) |
| 5   | -1.0 * 0.6 + varies by age |

More precisely, the true elasticity is `0.6 * ncd_elasticity + 0.4 * age_elasticity`, further modified by a 1.3x multiplier for PCW customers. Given this is a complex mixture, compute the mean `true_elasticity` from the dataset for each NCD band and compare to your GATE estimates.

```python
# True elasticity by NCD from the synthetic data
true_by_ncd = (
    df_renewals
    .group_by("ncd_years")
    .agg(pl.col("true_elasticity").mean().alias("true_elasticity"))
    .sort("ncd_years")
)

gate_ncd = est_forest.gate(df_renewals, by="ncd_years")

# Join and compare
comparison = true_by_ncd.join(gate_ncd, on="ncd_years")
print(comparison)
```

For each NCD band, state whether the recovered GATE is within the 95% confidence interval of the true elasticity.

### Part C: Interpreting CATE heterogeneity

The per-customer CATE values show substantial spread. Compute:

1. The 10th and 90th percentile of CATE values
2. The mean CATE for the top quartile of `last_premium` (most expensive policies)
3. The mean CATE for direct channel vs. PCW channel customers

Based on these numbers, describe in 2-3 sentences what the heterogeneity means for the pricing team's strategy.

<details>
<summary>Solution</summary>

### Part A

LinearDML assumes constant treatment effect (no heterogeneity across customers). Its ATE should be close to -2.0 and fits much faster than the causal forest. CausalForestDML allows the treatment effect to vary by customer characteristics and therefore provides both an ATE and per-customer CATE estimates.

The ATEs from the two models should be similar (within the confidence intervals of each other) if the ATE is approximately constant. If they differ substantially, the heterogeneous model is picking up genuine segment-level variation that the constant-effect model averages over.

### Part B

```python
gate_ncd = est_forest.gate(df_renewals, by="ncd_years")
comparison = true_by_ncd.join(gate_ncd, on="ncd_years")

print("NCD | True elasticity | Estimated | CI lower | CI upper | In CI?")
for row in comparison.iter_rows(named=True):
    in_ci = row["ci_lower"] <= row["true_elasticity"] <= row["ci_upper"]
    print(f"  {row['ncd_years']}  | {row['true_elasticity']:>8.3f}  | {row['elasticity']:>8.3f}  | "
          f"{row['ci_lower']:>8.3f}  | {row['ci_upper']:>8.3f}  | {'Yes' if in_ci else 'No'}")
```

On 50,000 observations, the CausalForestDML typically recovers the NCD-band GATEs within the 95% confidence intervals for most bands. The bands with fewest observations (extreme NCD levels) may have wider CIs that include the truth but are less precise.

### Part C

```python
cate_vals = est_forest.cate(df_renewals)

p10 = np.percentile(cate_vals, 10)
p90 = np.percentile(cate_vals, 90)
print(f"CATE p10: {p10:.3f}  p90: {p90:.3f}")

# By premium quartile
last_prem = df_renewals["last_premium"].to_numpy()
q75_prem = np.percentile(last_prem, 75)
top_quartile_mask = last_prem > q75_prem
print(f"CATE for top premium quartile: {cate_vals[top_quartile_mask].mean():.3f}")
print(f"CATE for bottom 75%:           {cate_vals[~top_quartile_mask].mean():.3f}")

# By channel
pcw_mask = df_renewals["channel"].to_numpy() == "pcw"
direct_mask = df_renewals["channel"].to_numpy() == "direct"
print(f"CATE for PCW channel:    {cate_vals[pcw_mask].mean():.3f}")
print(f"CATE for direct channel: {cate_vals[direct_mask].mean():.3f}")
```

The heterogeneity tells the pricing team: a single portfolio-average elasticity misrepresents the book. High-premium PCW customers are substantially more elastic (larger negative CATE) than low-premium direct customers. If you apply a uniform 5% price increase, you will lose a disproportionate share of your PCW book while barely affecting direct customers. The optimal strategy is different prices for different segments - which is exactly what the per-policy optimiser in Part 12 of the tutorial does.

</details>

---

## Exercise 7: Building and using the demand curve

**Reference:** Tutorial Part 11

**Estimated time:** 20 minutes

The pricing director asks: "if we raise renewal prices by 10% across the board, what happens to our renewal rate and our total profit?" Use the demand curve to answer this.

### Part A: Compute the demand curve

Use the fitted `est_forest` model from Exercise 6 (or refit if needed). Compute the portfolio demand curve over a range from -20% to +30%:

```python
demand_df = demand_curve(
    est_forest,
    df_renewals,
    price_range=(-0.20, 0.30, 50),
)
print(demand_df)
```

### Part B: Answer the director's question

Extract from the demand curve the predicted renewal rate and expected profit per policy at approximately +10% price change.

```python
row_10pct = demand_df.filter(
    (pl.col("pct_price_change") - 0.10).abs() < 0.01
).head(1)
print(row_10pct)
```

Compare this to the current (0% change) position. Write a one-paragraph summary suitable for a pricing committee slide:
- Current renewal rate and profit per policy at 0% change
- Predicted renewal rate and profit per policy at +10%
- The trade-off in plain English

### Part C: Finding the optimal portfolio price change

Which price change in the demand curve maximises expected profit per policy? Is the ENBP constraint the reason this is not the same as the price that maximises expected revenue? Explain in 2 sentences.

### Part D: Sensitivity to the ATE estimate

The ATE point estimate has a 95% confidence interval. Compute the demand curve using the lower and upper bounds of the CI to understand the range of outcomes:

```python
ate_point, lb, ub = est_forest.ate()

# We would need to refit the model with different ATEs for this,
# but we can approximate by scaling the CATE values.
# This gives a sense of the uncertainty band.

# At +10% price change, what range of renewal rates is consistent
# with the CI?
delta_log = np.log(1.10)  # 10% price increase

# Using point estimate
ate_effect_point = ate_point * delta_log
# Using CI bounds
ate_effect_lower = lb * delta_log
ate_effect_upper = ub * delta_log

baseline_renewal = df_renewals["renewed"].mean()
print(f"ATE effect of +10% at point estimate: {ate_effect_point:.4f}")
print(f"  -> Renewal rate change: {ate_effect_point*100:.2f}pp")
print(f"  -> Predicted renewal rate: {baseline_renewal + ate_effect_point:.3f}")
print(f"Lower CI: renewal rate {baseline_renewal + ate_effect_lower:.3f}")
print(f"Upper CI: renewal rate {baseline_renewal + ate_effect_upper:.3f}")
```

Is the uncertainty in the renewal rate prediction large enough to change the commercial recommendation?

<details>
<summary>Solution</summary>

### Part B

At 0% price change: the current renewal rate is the observed rate in the data (around 72-75%). The expected profit per policy is the average of (last_premium - tech_prem) * observed_renewal.

At +10% price change: renewal rate should be lower (more lapses from higher price), but margin per policy is higher. The net effect on profit depends on the elasticity.

Pricing committee summary: "Our demand model estimates that a portfolio-wide 10% price increase would reduce our renewal rate from approximately 73% to approximately 70%, a loss of around 3 percentage points. However, the higher margin on retained policies more than offsets the volume loss, increasing expected profit per policy from £X to £Y. We recommend targeting the portfolio-optimal price change of Z%, which the model identifies as the profit peak. Note this analysis is subject to the DML confidence interval uncertainty quantified in the appendix."

### Part C

The profit-maximising price change (from the demand curve) is typically positive - insurers tend to be slightly below the profit-optimal price because they prioritise volume. The revenue-maximising price (maximising price x renewal rate) is lower than the profit-maximising price, because at very high prices the renewal rate falls so much that revenue falls even though price is higher.

The ENBP constraint may not be the binding factor at the portfolio-average level: the profit peak from the demand curve may already be below the average ENBP headroom. But for individual customers (especially short-tenure ones on PCW), the ENBP constraint may prevent charging the individual-optimal price. The per-policy optimiser in Part 12 of the tutorial captures this.

### Part D

The uncertainty band around the renewal rate prediction at +10% price change is typically 1-3 percentage points wide. Whether this changes the commercial recommendation depends on the decision. If the profit-optimal price change is +10% but the CI covers +8% to +12%, the direction of the recommendation (raise prices) is robust. If the CI crosses the breakeven point (where profit at +10% equals profit at 0%), the recommendation would be to raise prices cautiously and monitor actual lapse rates. The demand curve does not make the decision; it narrows the set of plausible outcomes and makes the trade-offs explicit.

</details>

---

## Exercise 8: The ENBP optimisation in practice

**Reference:** Tutorial Parts 12-13

**Estimated time:** 25 minutes

This exercise runs the full FCA-compliant pricing optimisation and explores what it does with different customer segments.

### Part A: Run the optimiser

Use the `est_forest` model from Exercise 6 and run:

```python
opt = RenewalPricingOptimiser(
    est_forest,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,
)
priced_df = opt.optimise(df_renewals, objective="profit")
```

### Part B: Segment analysis

Compute the mean optimal price, mean ENBP headroom, and mean expected profit by NCD band:

```python
segment_summary = (
    priced_df
    .group_by("ncd_years")
    .agg([
        pl.col("optimal_price").mean().alias("mean_optimal_price"),
        pl.col("enbp_headroom").mean().alias("mean_enbp_headroom"),
        pl.col("expected_profit").mean().alias("mean_expected_profit"),
        pl.col("predicted_renewal_prob").mean().alias("mean_renewal_prob"),
        pl.len().alias("n"),
    ])
    .sort("ncd_years")
)
print(segment_summary)
```

For which NCD bands is the ENBP constraint most binding (lowest headroom)?

### Part C: Comparing profit vs. retention objectives

Re-run the optimiser with `objective="retention"` and compare the results:

```python
priced_retention = opt.optimise(df_renewals, objective="retention")
```

Compute the difference in mean optimal price and mean expected profit between the two objectives across the full portfolio:

```python
print("Profit objective - mean optimal price:  ",
      priced_df["optimal_price"].mean().round(2))
print("Retention objective - mean optimal price:",
      priced_retention["optimal_price"].mean().round(2))
print()
print("Profit objective - mean expected profit:  ",
      priced_df["expected_profit"].mean().round(2))
print("Retention objective - mean expected profit:",
      priced_retention["expected_profit"].mean().round(2))
```

Write 2-3 sentences describing the trade-off between the two objectives and when a firm might prefer the retention objective.

### Part D: Run the compliance audit

```python
audit = opt.enbp_audit(priced_df)
n_breaches = int((audit["compliant"] == False).sum())
print(f"Breaches: {n_breaches} of {len(audit)}")
```

If there are zero breaches (expected, since the optimiser enforces ENBP), print the five policies with the smallest ENBP headroom. These are the policies where the ENBP constraint was most nearly binding and where a data quality error in the ENBP column would be most likely to cause a breach.

<details>
<summary>Solution</summary>

### Part B

NCD=0 customers (no claims discount, typically young drivers) should show the lowest ENBP headroom because: their true elasticity is highest, so the profit-optimal price is not much above the technical floor; and their ENBP may be relatively constrained. However, the ENBP headroom depends on the relationship between the offered price and the new business equivalent - this depends on how the ENBP is calculated in the synthetic DGP.

High-NCD customers (NCD=5) should show the highest ENBP headroom because the profit-optimal price can be quite close to the ENBP ceiling for inelastic customers (their price sensitivity is low, so charging towards the ceiling barely reduces their renewal probability), but the optimiser can get there.

### Part C

```python
profit_mean_price = priced_df["optimal_price"].mean()
retention_mean_price = priced_retention["optimal_price"].mean()
print(f"Price difference: £{profit_mean_price - retention_mean_price:.2f}")
```

The retention objective sets prices at the technical premium floor (minimum price = maximum renewal probability). This maximises the probability that each customer renews, at the cost of accepting the lowest possible margin. A firm might prefer this objective if it is in a growth phase prioritising market share over margin, if it is trying to rebuild NCD capital after a period of high lapses, or if it is subject to a regulatory agreement requiring it to demonstrate customer fairness through low prices.

The profit objective sets prices at the level where the margin-volume trade-off is optimised for each customer individually. This is the right objective for a firm trying to maximise the total economic value of its renewal book.

### Part D

```python
# Policies closest to the ENBP ceiling
audit_sorted = audit.sort("margin_to_enbp")
print("Five policies with smallest ENBP headroom:")
print(audit_sorted.head(5).select(["policy_id", "offered_price", "enbp", "margin_to_enbp"]))
```

These borderline policies are the ones that a data quality review should prioritise. If the ENBP calculation has an error of even £1-2 for these policies, they might flip into non-compliance. In a real pricing process, policies within £5 of the ENBP ceiling would typically be manually reviewed before the prices are issued.

</details>

---

## Exercise 9: Presenting results to stakeholders

**Reference:** Tutorial Parts 10-14

**Estimated time:** 30 minutes

Pricing actuaries spend as much time communicating results as computing them. This exercise focuses on turning the demand model output into a presentation that a commercial director can act on.

### Part A: The one-page summary table

Produce a single summary table suitable for a pricing committee slide. It should contain the following columns:

- NCD band (0, 1-2, 3-4, 5+)
- Policy count
- Mean true elasticity (known from synthetic data; in reality, this is your DML estimate)
- Implied renewal rate change from a 10% price increase (= ATE x log(1.10))
- Mean ENBP headroom from the optimised portfolio
- Mean expected profit change vs. current (from the optimiser output)

Group the NCD years into four bands:

```python
df_summary = priced_df.with_columns(
    pl.when(pl.col("ncd_years") == 0).then(pl.lit("NCD 0"))
    .when(pl.col("ncd_years").is_in([1, 2])).then(pl.lit("NCD 1-2"))
    .when(pl.col("ncd_years").is_in([3, 4])).then(pl.lit("NCD 3-4"))
    .otherwise(pl.lit("NCD 5+"))
    .alias("ncd_band")
)
```

### Part B: The commercial recommendation

Based on the demand curve and optimiser output from Exercises 7 and 8, write a 200-word briefing note for the pricing committee covering:

1. The overall portfolio-level optimal price change
2. The segments where the ENBP constraint is most binding
3. The risk to the recommendation (from the CI uncertainty in the elasticity estimate)
4. The compliance sign-off (ENBP audit: zero breaches)

Keep it tight. No hedging. State a recommendation and justify it.

### Part C: The FCA question

An FCA analyst reviewing your pricing process asks: "How do you ensure that your renewal pricing does not systematically penalise long-tenure customers?"

Write a 150-word response that:
- Explains what the ENBP check does
- Confirms that the model does not use lapse propensity to set prices above ENBP
- Describes the audit trail available (MLflow run ID, per-policy audit table)

<details>
<summary>Solution</summary>

### Part A

```python
gate_for_summary = est_forest.gate(df_renewals, by="ncd_years")
delta_10pct = np.log(1.10)

summary_table = (
    df_summary
    .group_by("ncd_band")
    .agg([
        pl.len().alias("n_policies"),
        pl.col("true_elasticity").mean().alias("mean_elasticity"),
        pl.col("enbp_headroom").mean().alias("mean_enbp_headroom"),
        pl.col("expected_profit").mean().alias("mean_expected_profit"),
    ])
    .with_columns(
        (pl.col("mean_elasticity") * delta_10pct * 100).round(2)
        .alias("renewal_rate_change_10pct_pp")
    )
    .sort("ncd_band")
)
print(summary_table)
```

### Part B

Example briefing note:

"The demand model estimates a portfolio-average price elasticity of -2.0 (95% CI: -2.1 to -1.9). At current prices, the profit-optimal portfolio-wide price change is approximately +7%, which the model predicts would reduce renewal rate by 2.1 percentage points while increasing expected profit per policy by £8.40.

ENBP constraints are most binding for NCD 0-2 customers on the PCW channel, where the technically-optimal prices would exceed ENBP in approximately 18% of cases. The optimiser caps all such policies at ENBP. The profit shortfall from the ENBP constraint is estimated at £2.10 per affected policy per year.

The main risk to this recommendation is estimation uncertainty in the elasticity: the 95% CI implies a renewal rate outcome range of 70.1% to 71.8% at +7% price change. Both bounds support the direction of the recommendation (raise prices). We recommend a phased implementation with active monitoring of actual lapse rates against the model's predictions in the first quarter post-implementation.

Compliance: the ENBP audit confirms zero per-policy breaches across all 50,000 renewal policies. Audit trail: MLflow run ID [see appendix], Delta table pricing.motor.enbp_audit_log."

### Part C

"Our renewal pricing process complies with FCA PS21/5 through the following controls. First, for every renewal policy, we calculate the equivalent new business price (ENBP) - the price a new customer with identical risk characteristics would be quoted today on the same channel. The ENBP is calculated by the underwriting system, not by the demand model.

Second, the renewal pricing optimiser enforces a hard constraint: no policy is offered a renewal price above its ENBP. This applies at the individual policy level, not on average. The per-policy compliance status is recorded in our audit log (pricing.motor.enbp_audit_log in Unity Catalog) with the run date, model version, and signoff actuary for every pricing cycle.

Third, the demand model's role is to identify the profit-maximising price below the ENBP ceiling. It does not use estimated lapse probability to justify a higher price. Inelastic customers are priced closer to the ENBP ceiling because the elasticity model suggests a higher price does not materially reduce their renewal probability - not because we have identified them as unlikely to complain. The distinction is: we use price sensitivity to optimise below the ceiling, not to exceed it."

</details>

---

## Exercise 10: End-to-end pipeline

**Reference:** All tutorial parts

**Estimated time:** 45-60 minutes

This is the capstone exercise. You will build the complete pipeline from data to ENBP-compliant prices, treating everything you know as a first-time builder without any scaffolding from the tutorial.

You are given a new portfolio: 30,000 renewal policies with a different DGP from the main dataset (seed=999). Your job is to run the full pipeline and deliver the compliance-ready output.

### Setup: your new portfolio

```python
df_new = make_renewal_data(n=30_000, seed=999, price_variation_sd=0.10)
df_new = df_new.with_columns(
    (1 - pl.col("renewed")).alias("lapsed")
)
print(f"New portfolio: {len(df_new):,} policies")
print(f"Renewal rate: {df_new['renewed'].mean():.1%}")
```

### Task 1: Diagnostic

Run the treatment variation diagnostic. If `weak_treatment` is True, stop and report why. If False, proceed.

### Task 2: Fit the elasticity model

Fit a `RenewalElasticityEstimator` with `cate_model="linear_dml"` (faster for this exercise). Report the ATE with the 95% confidence interval.

### Task 3: Compute GATEs

Report the GATE by channel. Which channel has the highest price elasticity?

### Task 4: Build the demand curve

Sweep from -20% to +20% price change. Find the profit-maximising price change and the renewal rate at that point.

### Task 5: Run the optimiser

Run `RenewalPricingOptimiser` with `objective="profit"`. Report:
- Mean optimal price
- Mean ENBP headroom
- Proportion of policies where ENBP constraint is binding (headroom < £1)

### Task 6: Compliance audit

Run `enbp_audit()`. Confirm zero breaches. If there are breaches, identify the cause and propose a fix.

### Task 7: Write the output to a simulated Delta table

Produce a final DataFrame with columns: `policy_id`, `optimal_price`, `enbp`, `enbp_headroom`, `compliant`, `expected_profit`, and `run_date`. Sort by `policy_id`.

```python
from datetime import date

run_date = str(date.today())
# Build the final output DataFrame here
```

<details>
<summary>Full Solution</summary>

```python
# Task 1: Diagnostic
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(df_new, treatment="log_price_change",
                                          confounders=confounders)
print(report.summary())
if report.weak_treatment:
    print("STOP: weak treatment problem. Do not proceed to DML fitting.")
    print("Remedies:", report.suggestions)
```

```python
# Task 2: Elasticity model
est_new = RenewalElasticityEstimator(
    cate_model="linear_dml",
    catboost_iterations=300,
    n_folds=5,
)
est_new.fit(df_new, outcome="renewed", treatment="log_price_change",
            confounders=confounders)

ate, lb, ub = est_new.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
```

```python
# Task 3: GATEs by channel
gate_channel = est_new.gate(df_new, by="channel")
print("GATE by channel:")
print(gate_channel)
# Most elastic channel = largest negative elasticity value
most_elastic = gate_channel.sort("elasticity").row(0, named=True)
print(f"\nMost elastic channel: {most_elastic['channel']} (elasticity {most_elastic['elasticity']:.3f})")
```

```python
# Task 4: Demand curve
demand_df_new = demand_curve(est_new, df_new, price_range=(-0.20, 0.20, 40))

# Profit-maximising price change
max_profit = demand_df_new.sort("predicted_profit", descending=True).row(0, named=True)
print(f"Profit-maximising price change: {max_profit['pct_price_change']*100:.1f}%")
print(f"Renewal rate at optimum: {max_profit['predicted_renewal_rate']*100:.1f}%")
print(f"Expected profit at optimum: £{max_profit['predicted_profit']:.2f}")
```

```python
# Task 5: Optimiser
opt_new = RenewalPricingOptimiser(
    est_new,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,
)
priced_new = opt_new.optimise(df_new, objective="profit")

print(f"Mean optimal price:  £{priced_new['optimal_price'].mean():.2f}")
print(f"Mean ENBP headroom:  £{priced_new['enbp_headroom'].mean():.2f}")

binding = (priced_new["enbp_headroom"] < 1.0).sum()
print(f"ENBP binding: {binding:,} of {len(priced_new):,} ({100*binding/len(priced_new):.1f}%)")
```

```python
# Task 6: Compliance audit
audit_new = opt_new.enbp_audit(priced_new)
n_breaches = int((audit_new["compliant"] == False).sum())
print(f"ENBP breaches: {n_breaches}")
if n_breaches == 0:
    print("All policies compliant with FCA ICOBS 6B.2")
```

```python
# Task 7: Final output
from datetime import date

run_date = str(date.today())

final_output = (
    priced_new
    .join(audit_new.select(["policy_id", "compliant"]), on="policy_id")
    .select([
        "policy_id",
        "optimal_price",
        "enbp",
        "enbp_headroom",
        "compliant",
        "expected_profit",
    ])
    .with_columns(pl.lit(run_date).alias("run_date"))
    .sort("policy_id")
)

print(f"Final output: {len(final_output):,} rows")
print(final_output.head(5))

# In production:
# spark.createDataFrame(final_output.to_pandas()).write.format("delta") \
#     .mode("append").saveAsTable("pricing.motor.renewal_prices_compliant")
```

The complete pipeline - diagnostic, DML fitting, demand curve, per-policy optimisation, compliance audit - takes about 10-15 minutes end-to-end on Databricks Free Edition for 30,000 policies. In production, the fitting step (DML) would be run weekly or monthly and the scoring step (optimise + audit) would run daily against the live renewal book.

</details>

---

## Reference card

Quick reference for the APIs used in these exercises.

**Conversion model:**
```python
ConversionModel(base_estimator="logistic"|"catboost",
                feature_cols=[...], rank_position_col="rank_position",
                cat_features=[...])
.fit(df) .predict_proba(df) .oneway(df, "feature") .summary()
```

**Retention model:**
```python
RetentionModel(model_type="logistic"|"catboost",
               outcome_col="lapsed", price_change_col="log_price_change",
               feature_cols=[...], cat_features=[...])
.fit(df) .predict_proba(df) .predict_renewal_proba(df)
.price_sensitivity(df) .oneway(df, "feature")
```

**Diagnostic:**
```python
ElasticityDiagnostics().treatment_variation_report(
    df, treatment="log_price_change", confounders=[...])
# -> TreatmentVariationReport: .weak_treatment, .variation_fraction, .summary()
```

**Elasticity estimator (insurance-demand):**
```python
ElasticityEstimator(outcome_col="converted", treatment_col="log_price_ratio",
                    feature_cols=[...], n_folds=5)
.fit(df) .summary() .elasticity_ .elasticity_ci_ .sensitivity_analysis()
```

**Elasticity estimator (insurance-elasticity):**
```python
RenewalElasticityEstimator(cate_model="causal_forest"|"linear_dml",
                            n_estimators=200, catboost_iterations=500, n_folds=5)
.fit(df, outcome="renewed", treatment="log_price_change", confounders=[...])
.ate() .cate(df) .gate(df, by="column")
```

**Optimiser:**
```python
RenewalPricingOptimiser(est, technical_premium_col="tech_prem",
                         enbp_col="enbp", floor_loading=1.0)
.optimise(df, objective="profit"|"retention")
.enbp_audit(priced_df)
```

**Demand curve:**
```python
demand_curve(estimator, df, price_range=(-0.25, 0.25, 50))
# -> polars DataFrame: pct_price_change, predicted_renewal_rate, predicted_profit
```
