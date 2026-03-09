## Part 8: Causal elasticity estimation with DML

Now we fit the DML model. The `insurance-demand` library's `ElasticityEstimator` class handles the whole pipeline: building the nuisance models, cross-fitting, and extracting the elasticity estimate with confidence intervals.

### The treatment variable for conversion elasticity

For conversion data, the treatment variable is `log_price_ratio` = log(quoted_price / technical_premium). This measures the commercial loading above the technical price. We use this rather than the absolute quoted price because the absolute price is dominated by risk class - a £1,200 quote for a young driver and a £1,200 quote for a 45-year-old mean completely different things commercially.

The log_price_ratio column is already in the conversion dataset. Let us verify:

```python
%md
## Part 8: DML elasticity estimation (conversion)
```

```python
print(df_quotes.select(["log_price_ratio", "quoted_price", "technical_premium"]).describe())
```

A `log_price_ratio` of 0.0 means the quoted price equals the technical premium. Positive values mean we are loading above technical (profitable per policy but lower conversion). Negative values mean we are pricing below technical (volume-positive but margin-negative).

### Fitting the conversion elasticity model

```python
from insurance_demand import ElasticityEstimator

est_conversion = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    outcome_model="catboost",
    treatment_model="catboost",
    heterogeneous=False,
)

print("Fitting DML elasticity estimator... (this takes 3-5 minutes)")
est_conversion.fit(df_quotes)

print("\n=== DML Elasticity Summary ===")
print(est_conversion.summary())
```

The DML fit involves running 5-fold cross-fitting for both the outcome and treatment nuisance models. On 150,000 records with CatBoost nuisance models, this takes 3-5 minutes on Databricks Free Edition. Do not interrupt it.

The summary output looks like:

```
   parameter  estimate  std_error  ci_lower_95  ci_upper_95 treatment          outcome  n_folds
0  price_elasticity    -2.03       0.04        -2.11        -1.95  log_price_ratio  converted       5
```

The estimate should be close to the true elasticity of -2.0. This is what DML recovers: a causal estimate that is not contaminated by the risk composition effect.

Now compare to what the naive logistic regression gives:

```python
print(f"True elasticity (DGP):     {df_quotes['true_elasticity'].mean():.3f}")
print(f"DML estimate:              {est_conversion.elasticity_:.3f}")
print(f"DML 95% CI:                [{est_conversion.elasticity_ci_[0]:.3f}, {est_conversion.elasticity_ci_[1]:.3f}]")

# Naive logistic coefficient for comparison
logistic_summary = conv_logistic.summary()
naive_coef = logistic_summary.loc[logistic_summary["feature"] == "log_price_ratio", "coefficient"].values[0]
print(f"Naive logistic coefficient: {naive_coef:.3f}")
print(f"Bias in naive estimate:     {abs(naive_coef - df_quotes['true_elasticity'].mean()):.3f}")
```

The DML estimate should be very close to -2.0. The naive logistic will typically be more negative - say -2.4 to -2.8 - because it has not removed the confounding. The difference is the bias from risk composition.

### Interpreting the elasticity estimate

The PLR estimator runs OLS of the outcome residual on the treatment residual. The outcome is the raw binary conversion indicator, not its logit. The coefficient is therefore on the probability scale: a linear probability model coefficient.

The DML coefficient of -2.0 means: a 1-unit increase in log_price_ratio reduces the probability of conversion by 2.0 percentage points. No logistic transformation is involved. For a 10% price increase:

- A 10% increase in loading corresponds to a log change of approximately 0.095.
- Reduction in conversion probability: 2.0 x 0.095 = 0.19 percentage points -- call it 0.2 percentage points.
- At a base conversion rate of 12%, that is a 1.7% relative reduction in conversion.

On 150,000 quotes per year, 0.2pp fewer converts to roughly 255 fewer policies. That is a small absolute effect but meaningful at scale.

### Sensitivity analysis

How confident are we in this estimate? Could unobserved confounding (features we did not include) be driving the result?

```python
sensitivity = est_conversion.sensitivity_analysis()
if sensitivity is not None:
    print(sensitivity)
```

The sensitivity analysis reports how large unobserved confounding would need to be to drive the elasticity to zero. If the result shows that you would need an unobserved confounder with an R-squared of 0.40 to overturn the finding, the estimate is robust. If it says 0.03, you should be cautious.