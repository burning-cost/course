## Part 5: The shrinkage plot

The shrinkage plot is the chart that earns credibility modelling its budget. It shows, visually, what credibility is doing: pulling thin cells toward the portfolio mean and leaving dense cells close to their own experience.

```python
%md
## Part 5: Shrinkage plot
```

```python
# Build the data for the shrinkage plot
# We need: observed rate, credibility estimate, exposure (for sizing), Z (for colour)
plot_data = bs_results.join(
    dist_totals.select(["postcode_district", "total_earned_years"]),
    on="postcode_district",
    how="inner",
)

obs_rates = plot_data["obs_mean"].to_numpy()
cred_ests = plot_data["credibility_estimate"].to_numpy()
exposures = plot_data["total_earned_years"].to_numpy()
z_vals = plot_data["Z"].to_numpy()

grand_mean = bs["grand_mean"]

# Point sizes: log exposure, scaled to readable marker sizes
log_exp = np.log1p(exposures)
sizes = 15 + 120 * (log_exp - log_exp.min()) / (log_exp.max() - log_exp.min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    obs_rates,
    cred_ests,
    s=sizes,
    c=z_vals,
    cmap="RdYlGn",
    alpha=0.7,
    edgecolors="grey",
    linewidths=0.3,
    vmin=0, vmax=1,
)

# 45-degree line: perfect concordance between observed and credibility estimate
# Dense cells (high Z) should sit near this line
all_rates = np.concatenate([obs_rates, cred_ests])
rate_min = all_rates.min() * 0.8
rate_max = all_rates.max() * 1.1
ax.plot([rate_min, rate_max], [rate_min, rate_max],
        "k--", alpha=0.3, lw=1.5, label="No shrinkage (observed = estimate)")

# Horizontal line at grand mean: thin cells (low Z) should sit near this line
ax.axhline(grand_mean, color="steelblue", linestyle=":", alpha=0.6, lw=1.5,
           label=f"Grand mean = {grand_mean:.3f}")

plt.colorbar(sc, label="Credibility factor Z  (green = high Z, red = low Z)")

ax.set_xlabel("Observed claim frequency", fontsize=12)
ax.set_ylabel("Credibility-weighted estimate", fontsize=12)
ax.set_title("Bühlmann-Straub shrinkage plot\n(point size ∝ log exposure; colour = Z)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Plots observed rates on the x-axis against credibility estimates on the y-axis. Points near the 45-degree line are districts whose credibility estimates are close to their observed rate — these are dense districts with high Z. Points near the horizontal grand mean line are districts that have been pulled strongly toward the portfolio mean — these are thin districts with low Z.

**Run this cell.**

**What you should see:** A scatter plot where:
- Large green points (dense districts, high Z) cluster near the 45-degree dashed line
- Small red points (thin districts, low Z) cluster near the horizontal blue dotted line at the grand mean
- No point at an extreme observed rate has its credibility estimate also at that extreme, unless it has high exposure justifying it

This is the chart to show a pricing committee. It demonstrates in one image that the model is doing the right thing: trusting dense districts' experience and pulling thin districts back toward safety.