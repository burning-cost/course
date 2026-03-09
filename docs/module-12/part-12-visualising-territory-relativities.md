## Part 12: Visualising territory relativities

### Bar chart of relativities

The `plot_relativities` function shows the top and bottom areas by relativity with their credibility intervals.

```python
from insurance_spatial.plots import plot_relativities

fig = plot_relativities(
    rels,
    title="BYM2 Territory Relativities (synthetic 10x10 grid)",
    n_areas=30,
)
plt.show()
```

The error bars are the 95% credibility intervals. Notice that high-risk areas (red bars) in sparse grid cells have wider intervals than those in dense cells. This is the key difference from a non-Bayesian approach: the uncertainty is explicit and per-area.

### Grid choropleth

For the synthetic grid, we can visualise the territory relativities as a heatmap:

```python
# Extract relativities into a grid
rel_values = np.array(rels.sort("area")["relativity"].to_list())
# The areas are sorted alphabetically: r0c0, r0c1, ..., r9c9
# Reshape into grid
rel_grid = rel_values.reshape(NROWS, NCOLS)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True effects (known because synthetic)
true_rel_grid = np.exp(true_log_effect - true_log_effect.mean()).reshape(NROWS, NCOLS)
im0 = axes[0].imshow(true_rel_grid, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, origin="upper")
axes[0].set_title("True relativities (known; synthetic data)")
plt.colorbar(im0, ax=axes[0])

# BYM2 posterior mean relativities
im1 = axes[1].imshow(rel_grid, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, origin="upper")
axes[1].set_title("BYM2 posterior mean relativities")
plt.colorbar(im1, ax=axes[1])

# Naive O/E relativities (no smoothing)
naive_freq = claims / exposure
naive_rel = naive_freq / (claims.sum() / exposure.sum())
naive_grid = naive_rel.reshape(NROWS, NCOLS)
im2 = axes[2].imshow(naive_grid, cmap="RdYlGn_r", vmin=0.7, vmax=1.5, origin="upper")
axes[2].set_title("Naive O/E relativities (unsmoothed)")
plt.colorbar(im2, ax=axes[2])

plt.suptitle("Territory relativity estimation: BYM2 vs. naive O/E", y=1.02)
plt.tight_layout()
plt.show()
```

The three panels show the ground truth, the BYM2 estimate, and the naive (unsmoothed) estimate. BYM2 should be visibly smoother than naive O/E and visibly closer to the true pattern -- especially for sparse areas where naive O/E is very noisy.

### Comparing BYM2 to naive O/E numerically

```python
# Mean absolute error vs. truth
bym2_error = np.abs(rel_values - true_rel_grid.ravel()).mean()
naive_error = np.abs(naive_rel - true_rel_grid.ravel()).mean()

print(f"BYM2 MAE vs. truth:  {bym2_error:.4f}")
print(f"Naive O/E MAE:        {naive_error:.4f}")
print(f"Improvement:          {(1 - bym2_error/naive_error)*100:.1f}%")
```

On most runs with the fixed seed, BYM2 reduces the mean absolute error by 20--40% compared to naive O/E. The improvement is larger when there are more sparse areas.