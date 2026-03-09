## Part 9: Generating and interpreting prediction intervals

In a new cell:

```python
%md
## Part 9: Prediction intervals
```

```python
# 90% intervals: will cover the true outcome >= 90% of the time
intervals_90 = cp.predict_interval(X_test, alpha=0.10)

# 95% intervals: more conservative - for minimum premium floor applications
intervals_95 = cp.predict_interval(X_test, alpha=0.05)

# 80% intervals: used later in the practical minimum premium hybrid
intervals_80 = cp.predict_interval(X_test, alpha=0.20)

# Examine the structure
print("Output columns:", intervals_90.columns)
print("\nFirst 5 rows:")
print(intervals_90.head(5))
```

**What you should see:**

```
Output columns: ['point', 'lower', 'upper']

First 5 rows:
┌──────────┬──────────┬──────────┐
│ point    ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      │
│ f64      ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╡
│ xxx.xx   ┆ 0.00     ┆ xxx.xx   │
...
```

The `lower` column is clipped at zero because insurance losses cannot be negative. Many rows will have `lower = 0.0` - this is correct for zero-inflated Tweedie data where a substantial fraction of policies will have no claims.

Now examine the interval widths:

```python
widths_90 = intervals_90["upper"] - intervals_90["lower"]
point_est = intervals_90["point"].to_numpy()

print("90% Interval width distribution:")
print(f"  Min:    {widths_90.min():.2f}")
print(f"  Median: {widths_90.median():.2f}")
print(f"  Mean:   {widths_90.mean():.2f}")
print(f"  90th percentile: {widths_90.quantile(0.90):.2f}")
print(f"  Max:    {widths_90.max():.2f}")

# Key ratio: how much do intervals scale with risk level?
rel_widths = widths_90.to_numpy() / np.clip(point_est, 1e-6, None)
print(f"\nRelative width (interval width / point estimate):")
print(f"  Median: {np.median(rel_widths):.2f}")
print(f"  90th percentile: {np.quantile(rel_widths, 0.90):.2f}")
print(f"  Max:    {rel_widths.max():.2f}")
```

**Interpreting the output:** with `pearson_weighted` intervals, the absolute width scales roughly in proportion to the point estimate. A risk with a £1,000 point estimate will have an approximately 10x wider absolute interval than a risk with a £100 point estimate. This is correct - the genuine uncertainty is larger for larger risks. The relative width (interval / point estimate) should be more stable across risks, though it will be wider for thin-cell risks where the model is uncertain.