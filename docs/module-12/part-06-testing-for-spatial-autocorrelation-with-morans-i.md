## Part 6: Testing for spatial autocorrelation with Moran's I

Before fitting any model, we test whether spatial autocorrelation is actually present in the residuals. If it is not, spatial smoothing will not help and we are better off with a simpler approach.

### Computing the observed-to-expected ratio

The input to Moran's I should be residuals from a non-spatial model -- the part of the geographic variation that we cannot explain with non-spatial covariates. In a two-stage pipeline, this is the sector-level observed-to-expected ratio from the base model.

For now, we use the raw observed frequency relative to the portfolio mean, which is equivalent to an intercept-only non-spatial model:

```python
# Compute log O/E ratio per area
# O = observed claims; E = expected claims under null (uniform frequency)
portfolio_freq = claims.sum() / exposure.sum()
expected_null = exposure * portfolio_freq
log_oe = np.log((claims + 0.5) / (expected_null + 0.5))
# The +0.5 is a half-claim correction for zero-claim areas (Haldane's correction)

# Add to DataFrame for reference
df = df.with_columns(
    pl.Series("log_oe", log_oe.tolist())
)

print(f"Log O/E summary:")
print(f"  Mean:   {log_oe.mean():.4f}  (should be near zero)")
print(f"  SD:     {log_oe.std():.4f}")
print(f"  Min:    {log_oe.min():.4f}")
print(f"  Max:    {log_oe.max():.4f}")
```

### Running the Moran's I test

```python
from insurance_spatial.diagnostics import moran_i

test = moran_i(log_oe, adj, n_permutations=999)

print(f"Moran's I:       {test.statistic:.4f}")
print(f"Expected I:      {test.expected:.4f}")
print(f"Z-score:         {test.z_score:.2f}")
print(f"p-value:         {test.p_value:.4f}")
print(f"Significant:     {test.significant}")
print()
print(test.interpretation)
```

Because we generated data with a deliberate north-south spatial gradient, Moran's I should be significantly positive. The z-score tells you how many standard deviations the observed I is above the permutation-based null distribution. A z-score above 3.0 is strong evidence of spatial structure.

### Visualising the raw spatial pattern

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw observed frequency per area
freq_grid = (claims / exposure).reshape(NROWS, NCOLS)
im0 = axes[0].imshow(freq_grid, cmap="RdYlGn_r", origin="upper")
axes[0].set_title("Observed claim frequency (raw)")
plt.colorbar(im0, ax=axes[0])
for r in range(NROWS):
    for c in range(NCOLS):
        axes[0].text(c, r, f"{freq_grid[r,c]:.2f}", ha="center", va="center",
                     fontsize=6, color="black")

# Right: log O/E
logoe_grid = log_oe.reshape(NROWS, NCOLS)
im1 = axes[1].imshow(logoe_grid, cmap="RdYlGn_r", origin="upper")
axes[1].set_title(f"Log O/E (Moran's I = {test.statistic:.3f}, p = {test.p_value:.3f})")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
```

The spatial structure should be visible in both plots: darker (higher risk) cells concentrated at the top, lighter at the bottom. The raw frequency map will be noisier than the log O/E map because small areas dominate the visual.

### What if Moran's I is not significant?

If the test is not significant (p > 0.05), spatial smoothing is not warranted by the data. In that case:

- Use Bühlmann-Straub credibility per sector (Module 6) instead
- Or collapse to district level where you may have more data per area
- Do not fit BYM2 just because it is more sophisticated

We include this caution because significance testing before model selection is good practice, not pedantry. The BYM2 model will still fit even without spatial structure -- it will return rho near zero and the estimates will converge to the non-spatial model. But it is slower and harder to explain. If the data say "no spatial structure", believe the data.