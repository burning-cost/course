## Part 9: Heterogeneous elasticity with CausalForestDML

The global ATE tells you the portfolio-average elasticity. But the actual commercial question is more granular: are PCW customers more elastic than direct customers? Are young drivers more elastic than older drivers? Does elasticity vary by NCD band?

The answer matters for pricing decisions. If NCD-0 customers (typically young, first year of driving) are much more elastic than NCD-5 customers, you should price the two groups differently. A uniform price increase hits the elastic group hard and barely affects the inelastic group.

For this analysis we switch to the `insurance-elasticity` library's `RenewalElasticityEstimator`, which wraps `econml`'s `CausalForestDML` to estimate per-customer heterogeneous treatment effects (CATE).

**Time note:** The CausalForestDML fit below takes 5--8 minutes on Databricks Free Edition. After submitting the cell, read the GATE interpretation section (Part 10) while the model trains -- the content is directly relevant to reading the output.

### Fitting the CausalForestDML model

```python
%md
## Part 9: Heterogeneous elasticity (CausalForestDML)
```

```python
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

est_renewal = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
)

print("Fitting CausalForestDML... (5-8 minutes on Databricks Free Edition)")
est_renewal.fit(
    df_renewals,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)

ate, lb, ub = est_renewal.ate()
print(f"\nATE:    {ate:.3f}")
print(f"95% CI: [{lb:.3f}, {ub:.3f}]")
print(f"True ATE: {df_renewals['true_elasticity'].mean():.3f}")
```

The CausalForestDML is more computationally intensive than the PLR approach. On 50,000 records it takes 5-8 minutes. The ATE should be close to -2.0.

### Per-customer CATE estimates

```python
# Per-customer estimated elasticity
cate_values = est_renewal.cate(df_renewals)

print("CATE distribution:")
print(f"  Mean:    {cate_values.mean():.3f}")
print(f"  Std:     {cate_values.std():.3f}")
print(f"  Min:     {cate_values.min():.3f}")
print(f"  Max:     {cate_values.max():.3f}")
```

The CATE values represent individual-level heterogeneity. A customer with CATE of -3.5 is highly elastic: even a modest price increase causes a large drop in their renewal probability. A customer with CATE of -0.8 is inelastic: price increases have little effect.

In the synthetic data, the true elasticity varies by NCD band and age band. A young driver with no NCD (NCD=0) has a true elasticity of about -3.5; an older driver with 5 years NCD has a true elasticity of about -1.0. The CausalForestDML should recover this pattern.

### Group average treatment effects (GATE) by segment

```python
# GATE by NCD years
gate_ncd = est_renewal.gate(df_renewals, by="ncd_years")
print("GATE by NCD years:")
print(gate_ncd)
```

The GATE table shows the average elasticity within each NCD group with confidence intervals. You should see a clear pattern: NCD=0 customers are much more elastic (larger negative value) than NCD=5 customers. This is the heterogeneity the CausalForestDML is designed to capture.

```python
# GATE by channel
gate_channel = est_renewal.gate(df_renewals, by="channel")
print("\nGATE by channel:")
print(gate_channel)
```

PCW customers should be more elastic than direct or broker customers. PCW customers are active price shoppers; direct customers have self-selected out of the comparison market.

```python
# GATE by age band
gate_age = est_renewal.gate(df_renewals, by="age_band")
print("\nGATE by age band:")
print(gate_age)
```

The pattern here is: younger customers (17-24) are most elastic, older customers (65+) are least elastic. This matches the known DGP in the synthetic data.

Compare the recovered GATEs to the true values from the DGP:

```python
# True elasticities by NCD (from the data generator documentation)
true_by_ncd = {0: -3.5, 1: -3.0, 2: -2.5, 3: -2.0, 4: -1.5, 5: -1.0}

print("\nValidation: recovered vs true elasticity by NCD")
print(f"{'NCD':>6} {'True':>8} {'Recovered':>12} {'Lower':>8} {'Upper':>8}")
for row in gate_ncd.iter_rows(named=True):
    ncd = row["ncd_years"]
    print(f"{ncd:>6} {true_by_ncd.get(ncd, 'N/A'):>8} {row['elasticity']:>12.3f} {row['ci_lower']:>8.3f} {row['ci_upper']:>8.3f}")
```

On 50,000 observations with sufficient price variation, the recovered GATEs should be within the 95% confidence intervals of the true values for all NCD groups.