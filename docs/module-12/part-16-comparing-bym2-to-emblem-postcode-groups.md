## Part 16: Comparing BYM2 to Emblem postcode groups

This is the practical payoff. We quantify the difference between BYM2 and k-means banding.

### Simulating Emblem banding on the synthetic data

```python
from sklearn.cluster import KMeans

# k-means on log O/E ratios -- the Emblem approach
K = 8  # number of territory bands
log_oe_for_km = log_oe.reshape(-1, 1)

km = KMeans(n_clusters=K, random_state=42, n_init=10)
km.fit(log_oe_for_km)
band_assignments = km.labels_

# Compute band relativities: mean O/E within each band
band_oe = np.zeros(K)
for k in range(K):
    mask = band_assignments == k
    band_oe[k] = (claims[mask].sum() / exposure[mask].sum()) / portfolio_freq

# Assign band relativity to each area
naive_band_rel = band_oe[band_assignments]

print(f"Number of bands: {K}")
print(f"Band relativities: {np.sort(band_oe)}")
print()
# Compare range of relativities
print(f"BYM2 relativity range:  [{rels['relativity'].min():.4f}, {rels['relativity'].max():.4f}]")
print(f"Band relativity range:  [{naive_band_rel.min():.4f}, {naive_band_rel.max():.4f}]")
```

### Visualising the comparison

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BYM2
im0 = axes[0].imshow(rel_grid, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, origin="upper")
axes[0].set_title("BYM2 (smoothed, sector-level)")
plt.colorbar(im0, ax=axes[0])

# k-means banding
band_grid = naive_band_rel.reshape(NROWS, NCOLS)
im1 = axes[1].imshow(band_grid, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, origin="upper")
axes[1].set_title(f"k-means banding (k={K})")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
```

The banded map has sharp discontinuities at band boundaries. Adjacent cells in different bands show discrete jumps. The BYM2 map is spatially smooth: risk changes gradually. Both map approximate the true underlying pattern, but BYM2 does it without introducing artificial discontinuities.

### Which is more accurate?

```python
bym2_mae = np.abs(rel_values - true_rel_grid.ravel()).mean()
band_mae  = np.abs(naive_band_rel - true_rel_grid.ravel()).mean()

print(f"BYM2 MAE vs. truth:    {bym2_mae:.4f}")
print(f"k-means banding MAE:   {band_mae:.4f}")

# Correlation with true relativities
bym2_corr = np.corrcoef(rel_values, true_rel_grid.ravel())[0, 1]
band_corr  = np.corrcoef(naive_band_rel, true_rel_grid.ravel())[0, 1]

print(f"BYM2 correlation:      {bym2_corr:.4f}")
print(f"k-means correlation:   {band_corr:.4f}")
```

BYM2 will be more accurate (lower MAE, higher correlation) in most runs. The advantage is most pronounced for sparse areas where k-means banding is dominated by noise.