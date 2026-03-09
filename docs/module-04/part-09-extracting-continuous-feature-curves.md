## Part 9: Extracting continuous feature curves

Continuous features - `ncd_years`, `vehicle_group`, and `driver_age` - cannot be aggregated by unique value. There are 69 distinct ages from 17 to 85; each appears only a few hundred times. Instead, we fit a smooth curve through the per-observation SHAP values.

In a new cell, type this and run it (Shift+Enter):

**What LOESS is:** LOESS (locally estimated scatterplot smoothing) is a non-parametric smoothing method that fits a weighted local regression at each point along the feature's range. For a given evaluation point, it weights nearby observations more heavily and distant observations less, then fits a low-degree polynomial through those weighted points. This produces a smooth curve without imposing a global functional form - useful for driver age, where the true effect is strongly non-linear (high at both extremes, flat in the middle) and any global polynomial would force an artificially symmetric shape.

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",
)

print(f"Age curve shape: {age_curve.shape}")
print(age_curve.head(5))
```

You will see a DataFrame with 100 rows, one for each evaluation point along the driver age range:

```bash
Age curve shape: (100, 4)
   feature_value  relativity  lower_ci  upper_ci
0           17.0       1.834       NaN       NaN
1           17.7       1.812       NaN       NaN
...
```

Note that `lower_ci` and `upper_ci` are NaN for LOESS curves. Confidence intervals on smoothed curves require bootstrap resampling, which is not yet implemented in the library. You have to present smoothed curves without formal CIs - be explicit about that when presenting.

Now plot the age curve. In a new cell, type this and run it (Shift+Enter):

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(age_curve["feature_value"], age_curve["relativity"], color="steelblue", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Driver age")
ax.set_ylabel("Relativity (base level normalisation)")
ax.set_title("Frequency relativity by driver age - GBM (LOESS smoothed)")
ax.set_ylim(0.5, 2.5)
plt.tight_layout()
display(fig)
```

You will see a plot with the relativity on the vertical axis and driver age on the horizontal. The curve should show:

- High relativities (above 1.5) for drivers aged 17-22
- Declining quickly through the mid-20s
- A relatively flat section from roughly 30 to 65
- A mild upward curve for drivers above 70

This U-shape matches the true DGP: young drivers and elderly drivers both have elevated frequency. The GLM with a linear age term cannot reproduce this shape - it will show either a weak negative slope or a flat line through the middle, depending on the portfolio mix. The GBM's non-linear age curve is one of the clearest illustrations of why GBMs outperform GLMs on motor data.

Now extract the vehicle group curve. In a new cell, type this and run it (Shift+Enter):

```python
vg_curve = sr.extract_continuous_curve(
    feature="vehicle_group",
    n_points=50,
    smooth_method="loess",
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(vg_curve["feature_value"], vg_curve["relativity"], color="firebrick", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("ABI vehicle group")
ax.set_ylabel("Relativity")
ax.set_title("Frequency relativity by vehicle group - GBM")
plt.tight_layout()
display(fig)
```

The vehicle group curve should increase roughly monotonically from group 1 (lowest risk) to group 50 (highest risk). The true DGP has a linear effect of `+0.01` per group unit, which means `exp(0.01 × (50 - 1)) = exp(0.49) ≈ 1.63` from group 1 to group 50. The GBM may find a slightly non-linear curve, especially at the extremes where data is sparser.