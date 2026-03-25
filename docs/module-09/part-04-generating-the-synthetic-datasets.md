## Part 4: Generating the synthetic dataset

### The data-generating process

We work with a synthetic UK motor renewal dataset where the true elasticity is known. This lets us verify that the estimator recovers the right answer before applying it to real data where we cannot check.

The data generator (`make_renewal_data`) creates a portfolio where:

- Quoted prices are a near-deterministic function of risk factors, plus a small exogenous shock required for identification
- True elasticity varies by NCD band: NCD=0 customers are most elastic (−3.5), NCD=5 least elastic (−1.0)
- True elasticity varies by age: 17–24 most elastic (−3.0), 65+ least elastic (−1.2)
- PCW customers are 30% more elastic than direct or broker customers at the same NCD/age combination
- Renewal probability follows a logistic model with the true heterogeneous elasticity

The `price_variation_sd` parameter controls the standard deviation of the exogenous price shock around the re-rated technical change. At 0.08 (the default), DML has enough residual treatment variation. At 0.01 (simulated via `near_deterministic=True`), it does not — we demonstrate this in Part 5.

### Generating the dataset

```python
%md
## Part 4: Synthetic renewal dataset
```

```python
df = make_renewal_data(n=50_000, seed=42, price_variation_sd=0.08)

print(f"Shape:                 {df.shape}")
print(f"Renewal rate:          {df['renewed'].mean():.3f}")
print(f"Mean log price change: {df['log_price_change'].mean():.4f}")
print(f"Std log price change:  {df['log_price_change'].std():.4f}")
print()
print(df.select(["age", "ncd_years", "channel", "last_premium", "enbp",
                 "log_price_change", "renewed", "true_elasticity"]).head(5))
```

Expected output: renewal rate around 0.82–0.85, mean log price change around 0.05 (roughly 5% average price increase), standard deviation around 0.08.

### Understanding the key columns

| Column | Description |
|--------|-------------|
| `log_price_change` | log(offer_price / last_premium) — the treatment variable D |
| `renewed` | Binary renewal indicator — the outcome Y |
| `tech_prem` | Technical premium — the cost floor for the optimiser |
| `enbp` | Equivalent new business price — the FCA PS21/5 ceiling |
| `true_elasticity` | Ground truth per-customer CATE for validation (not available on real data) |

### Inspecting the ground-truth heterogeneity

The true elasticities are in the dataset. Before fitting any model, understand what we are trying to recover:

```python
# True average elasticity by NCD band
true_ncd = true_gate_by_ncd(df)
print("True elasticity by NCD band (DGP):")
print(true_ncd)
```

```python
# True average elasticity by age band
true_age = true_gate_by_age(df)
print("\nTrue elasticity by age band (DGP):")
print(true_age)
```

The NCD gradient runs from around −3.5 for NCD=0 to around −1.0 for NCD=5. The age gradient runs from around −3.0 for 17–24 to around −1.2 for 65+. These are the benchmarks we check our estimates against in Parts 7 and 8.

On real data you will not have `true_elasticity`. Fitting on synthetic data and comparing to ground truth is how you validate the methodology before trusting it on production data.
