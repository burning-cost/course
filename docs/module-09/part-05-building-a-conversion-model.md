## Part 5: The naive regression — why it fails

Before running the DML estimator, it is worth demonstrating the bias in the naive approach directly. The naive approach is a logistic regression of the renewal indicator on log price change plus risk features. It is what most teams run in practice. It gives a systematically biased elasticity estimate.

### Fitting the naive logistic model

```python
%md
## Part 5: Naive regression benchmark
```

```python
import statsmodels.formula.api as smf

# Encode categoricals as strings so statsmodels handles them
df_pd = df.with_columns([
    pl.col("channel").cast(pl.Utf8),
    pl.col("vehicle_group").cast(pl.Utf8),
    pl.col("region").cast(pl.Utf8),
]).to_pandas()

formula = (
    "renewed ~ log_price_change + age + ncd_years "
    "+ C(vehicle_group) + C(region) + C(channel)"
)
naive_logit = smf.logit(formula, data=df_pd).fit(disp=0)

price_coef = naive_logit.params["log_price_change"]
true_ate   = float(df["true_elasticity"].mean())

print(f"True ATE (DGP):         {true_ate:.3f}")
print(f"Naive logistic coef:    {price_coef:.3f}")
print(f"Absolute bias:          {abs(price_coef - true_ate):.3f}")
print(f"Relative bias:          {abs(price_coef - true_ate) / abs(true_ate) * 100:.1f}%")
```

The naive estimate will be more negative than the true ATE — the model attributes part of the risk-composition effect to the price effect. On UK motor renewal data with the DGP in this module, expect a bias of 20–40%. If you used this coefficient in Module 7's rate optimiser, you would believe your book is more elastic than it is, over-constrain price increases, and leave money on the table.

### Why conditioning on risk features is not enough

The natural objection: "but I did include age, NCD, vehicle group, and region — I am conditioning on the confounders." This does not fix the problem.

The issue is functional form. The logistic regression includes the confounders as main effects, but the price system has complex interactions between risk factors. The technical premium is a product of relativities:

```
tech_prem = base × f(age) × f(ncd) × f(vehicle) × f(region) × ...
```

Log-transforming gives an additive structure, but the mapping from risk features to price change also depends on the re-rating model applied that year, competitive adjustments by segment, and any manual overrides. A logistic regression with main effects does not capture these interactions. The residual — the price variation not explained by the linear risk feature combination — still contains variation correlated with the outcome through unmeasured interaction paths.

DML uses flexible machine learning models (CatBoost in our case) to estimate E[D|X] non-parametrically, capturing all these interactions. Only after removing E[D|X] does it use the residual for identification.

### Recording the naive estimate for comparison

```python
naive_coef = price_coef  # save for later comparison with DML
print(f"Naive estimate saved: {naive_coef:.3f}")
```

We will return to this comparison in Part 7 after the DML fit.
