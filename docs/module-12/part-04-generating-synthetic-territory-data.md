## Part 4: Generating synthetic territory data

We use a synthetic 10x10 grid of 100 areas to make the mechanics visible before moving to real postcode geography. The grid has a deliberate spatial structure: a high-risk band in the north and a low-risk band in the south, mimicking what you might see in a simplified UK book where urban risk concentrates in particular regions.

```python
rng = np.random.default_rng(seed=42)
NROWS, NCOLS = 10, 10
N = NROWS * NCOLS  # 100 areas

# True log-scale spatial effects
# North (rows 0-2): elevated risk; South (rows 7-9): reduced risk; Centre: near zero
row_idx = np.array([r for r in range(NROWS) for c in range(NCOLS)])
true_log_effect = np.where(row_idx <= 2, 0.35, np.where(row_idx >= 7, -0.25, 0.0))
# Add some smooth geographic noise plus area-level scatter
smooth_noise = 0.05 * np.sin(np.linspace(0, 2 * np.pi, N))
area_scatter = rng.normal(0, 0.08, N)
true_log_effect = true_log_effect + smooth_noise + area_scatter

# Exposure: policies per area (varies -- some areas are sparse)
exposure = rng.gamma(shape=2.5, scale=20.0, size=N).astype(int) + 5
# Base claim frequency: 8% before territory
base_freq = 0.08
expected_claims = exposure * base_freq * np.exp(true_log_effect)
# Observed claims: Poisson draw
claims = rng.poisson(expected_claims)

# Area labels
areas = [f"r{r}c{c}" for r in range(NROWS) for c in range(NCOLS)]

print(f"Areas:           {N}")
print(f"Total exposure:  {exposure.sum():,} policy-years")
print(f"Total claims:    {claims.sum():,}")
print(f"Overall freq:    {claims.sum() / exposure.sum():.4f}")
print(f"Areas zero clms: {(claims == 0).sum()}")
print(f"Sparse areas (<5 clms): {(claims < 5).sum()}")
```

Run this cell. You will see that despite only 100 areas, a meaningful fraction have zero or near-zero claims. This is the thin data problem in miniature. With 11,200 real postcode sectors, the sparsity is severe.

### Creating a Polars DataFrame for the area-level data

```python
df = pl.DataFrame({
    "area":     areas,
    "row":      row_idx.tolist(),
    "col":      [c for r in range(NROWS) for c in range(NCOLS)],
    "exposure": exposure.tolist(),
    "claims":   claims.tolist(),
    "true_log_effect": true_log_effect.tolist(),
})

# Observed frequency per area
df = df.with_columns(
    (pl.col("claims") / pl.col("exposure")).alias("obs_freq")
)

print(df.head(12))
```

The `true_log_effect` column exists only because this is synthetic data. In a real application, you do not know the true effects -- that is what the model will estimate.