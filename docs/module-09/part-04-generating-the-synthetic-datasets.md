## Part 4: Generating the synthetic datasets

We use the built-in data generators from both libraries. The generators produce datasets with known true elasticities, which means we can verify our estimates against the ground truth.

Add a markdown cell:

```python
%md
## Part 4: Generating the datasets
```

Then in a new code cell:

```python
# New business quotes: 150,000 records with true elasticity = -2.0
df_quotes = generate_conversion_data(n_quotes=150_000, seed=42)

print("=== Conversion dataset ===")
print(f"Shape:            {df_quotes.shape}")
print(f"Conversion rate:  {df_quotes['converted'].mean():.3f}")
print(f"True elasticity:  {df_quotes['true_elasticity'].mean():.3f}")
print(f"Channels:         {df_quotes['channel'].unique().sort().to_list()}")
print()
print("Price ratio stats:")
print(df_quotes.select(["price_ratio", "log_price_ratio"]).describe())
```

You should see something like:

```bash
=== Conversion dataset ===
Shape:            (150000, 17)
Conversion rate:  0.118
True elasticity:  -2.001
Channels:         ['direct', 'pcw_confused', 'pcw_ctm', 'pcw_go', 'pcw_msm']
```

The conversion rate of about 12% is realistic for a UK PCW motor insurer. On a PCW, most people compare 10-20 quotes and only one company wins each customer. A 12% conversion rate means you win roughly one in eight of the quotes you provide.

Now generate the renewal dataset:

```python
# Renewals: 50,000 records with heterogeneous true elasticity
df_renewals = make_renewal_data(n=50_000, seed=42)

print("=== Renewal dataset ===")
print(f"Shape:          {df_renewals.shape}")
print(f"Renewal rate:   {df_renewals['renewed'].mean():.3f}")
print(f"True ATE:       {df_renewals['true_elasticity'].mean():.3f}")
print()
print("Log price change distribution:")
print(df_renewals.select("log_price_change").describe())
```

The renewal rate should be around 72-75% and the true ATE around -2.0. The `log_price_change` column is the treatment variable: `log(offer_price / last_year_price)`. A value of 0.05 means a roughly 5% price increase.

In the next cell, look at the column names to understand the schema before building any models:

```python
print("Conversion columns:", df_quotes.columns)
print()
print("Renewal columns:", df_renewals.columns)
```

Key columns in the conversion dataset:
- `quoted_price`: the price we offered
- `technical_premium`: the risk model's output at quote time
- `log_price_ratio`: log(quoted_price / technical_premium) - this is the treatment variable for conversion elasticity
- `converted`: 1 if the customer bought, 0 if not
- `rank_position`: 1 if we were cheapest on the PCW, 2 if second cheapest, etc.
- `true_elasticity`: the ground truth (only in synthetic data; you will never have this in production)

Key columns in the renewal dataset:
- `log_price_change`: log(offer_price / last_year_price) - the treatment variable for renewal elasticity
- `renewed`: 1 if the customer renewed, 0 if they lapsed
- `tech_prem`: technical (cost-equivalent) premium
- `enbp`: equivalent new business price (the PS21/5 ceiling)
- `true_elasticity`: the ground truth, which varies by NCD and age band