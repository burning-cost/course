## Part 14: Lee diagrams

### What a Lee diagram shows

The Lee diagram (also called a loss-exposure plot) is a visualisation standard in reinsurance. It ranks risks by their destruction rate (loss/MPL) and plots:

- x-axis: cumulative proportion of risks (from lowest to highest destruction rate)
- y-axis: cumulative proportion of losses

If every risk had the same destruction rate, the diagram would be a straight 45-degree line. When losses are concentrated in a few risks (large partial losses or total losses), the Lee curve bows above the diagonal: 20% of risks might account for 70% of losses.

The fitted MBBEFD exposure curve overlaid on the Lee diagram is precisely G(x): for each fraction x of MPL, G(x) is the proportion of expected loss below that point. In the Lee diagram, x corresponds to the fraction of risks ranked below that destruction-rate level. The curve and the diagram are using the same x-axis.

```python
%md
## Part 14: Lee diagrams
```

```python
from insurance_ilf import lee_diagram

# Use the claims data from Part 8
losses = claims_df["loss_amount"].to_numpy()
mpl_arr = claims_df["mpl"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 6))

lee_diagram(losses=losses, mpl=mpl_arr, dist=fitted_dist, ax=ax)
ax.set_title("Lee Diagram: Observed losses vs fitted G(x)")
plt.tight_layout()
plt.show()
```

**Reading the diagram:**

The scatter of dots is the empirical Lee curve from actual claims. The red line is the theoretical G(x) from the fitted MBBEFD. The dashed diagonal is the reference for equal distribution.

A good fit shows the red line running through the centre of the scatter. If the empirical points systematically lie above the red line, the fitted curve is underestimating loss concentration (the tail is heavier than the model thinks). If they lie below, the model is overstating concentration.

For communication with underwriters, the Lee diagram is more immediately legible than a QQ plot. Show it to the underwriting team alongside the numerical goodness-of-fit tests.

```python
# Side-by-side: Lee diagram + exposure curve plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Lee diagram
lee_diagram(losses=losses, mpl=mpl_arr, dist=fitted_dist, ax=axes[0])
axes[0].set_title("Lee Diagram")

# Right: Fitted vs empirical exposure curve + standard curves
x_grid = np.linspace(0, 1, 500)
curves_to_plot = {"Y2": swiss_re_curve(2.0), "Y3": swiss_re_curve(3.0)}
for label, dist in curves_to_plot.items():
    axes[1].plot(x_grid, dist.exposure_curve(x_grid), "--", label=f"Swiss Re {label}", alpha=0.6)
axes[1].plot(x_grid, fitted_dist.exposure_curve(x_grid), "b-", linewidth=2,
             label=f"Fitted MBBEFD(g={fitted_dist.g:.1f}, b={fitted_dist.b:.1f})")
axes[1].scatter(ec_empirical.x_points, ec_empirical.g_values,
                s=15, color="orange", alpha=0.7, label="Empirical", zorder=5)
axes[1].set_xlabel("Fraction of MPL")
axes[1].set_ylabel("G(x)")
axes[1].set_title("Exposure Curve Comparison")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```