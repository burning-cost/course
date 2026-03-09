## Part 10: The shrinkage plot — Bayesian version

```python
%md
## Part 10: Bayesian shrinkage plot and uncertainty visualisation
```

```python
# Shrinkage plot comparing Bühlmann-Straub and Bayesian estimates
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

obs_rates = results["observed_rate"].to_numpy()
post_means = results["posterior_mean"].to_numpy()
bs_ests = bs_results.sort("postcode_district")["credibility_estimate"].to_numpy()
exposures_plot = results["earned_years"].to_numpy()

log_exp_p = np.log1p(exposures_plot)
sizes_p = 15 + 120 * (log_exp_p - log_exp_p.min()) / (log_exp_p.max() - log_exp_p.min() + 1e-9)

all_rates = np.concatenate([obs_rates, post_means, bs_ests])
rate_min = all_rates.min() * 0.8
rate_max = all_rates.max() * 1.1

# Left: Bühlmann-Straub
sc1 = axes[0].scatter(
    obs_rates, bs_ests,
    s=sizes_p, c=z_vals, cmap="RdYlGn", alpha=0.7,
    edgecolors="grey", linewidths=0.3, vmin=0, vmax=1,
)
axes[0].plot([rate_min, rate_max], [rate_min, rate_max], "k--", alpha=0.3, lw=1.5)
axes[0].axhline(bs["grand_mean"], color="steelblue", linestyle=":", alpha=0.6, lw=1.5)
axes[0].set_xlabel("Observed frequency")
axes[0].set_ylabel("Credibility estimate")
axes[0].set_title("Bühlmann-Straub")

# Right: Bayesian
sc2 = axes[1].scatter(
    obs_rates, post_means,
    s=sizes_p, c=z_bayes_approx, cmap="RdYlGn", alpha=0.7,
    edgecolors="grey", linewidths=0.3, vmin=0, vmax=1,
)
axes[1].plot([rate_min, rate_max], [rate_min, rate_max], "k--", alpha=0.3, lw=1.5)
axes[1].axhline(grand_mean_rate, color="steelblue", linestyle=":", alpha=0.6, lw=1.5)
axes[1].set_xlabel("Observed frequency")
axes[1].set_ylabel("Posterior mean frequency")
axes[1].set_title("Bayesian hierarchical (PyMC)")

plt.colorbar(sc2, ax=axes[1], label="Approximate Z  (green=high, red=low)")
plt.suptitle("Shrinkage comparison: Bühlmann-Straub vs Bayesian\n(size ∝ log exposure)", fontsize=12)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Side-by-side comparison of the B-S and Bayesian shrinkage. The patterns should look similar — both are applying partial pooling to the same data.

**Run this cell.**

**What you should see:** Two scatter plots with the same overall pattern. The Bayesian plot may show slightly more shrinkage for thin districts (lower z_bayes_approx for small points). Dense districts (large green points) should cluster near the 45-degree line in both plots. This visual confirmation tells you the two methods agree where they should.

### Uncertainty bands for individual districts

```python
# Plot credible intervals for the 20 districts with most evidence (densest)
# and 20 with least evidence (thinnest)
dense_20 = results.sort("earned_years", descending=True).head(20)
thin_20 = results.sort("earned_years").head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax, subset, title in [
    (axes[0], dense_20, "20 densest districts"),
    (axes[1], thin_20, "20 thinnest districts"),
]:
    districts_plot = subset["postcode_district"].to_list()
    obs = subset["observed_rate"].to_numpy()
    post = subset["posterior_mean"].to_numpy()
    lo = subset["lower_90"].to_numpy()
    hi = subset["upper_90"].to_numpy()
    x = np.arange(len(districts_plot))

    ax.barh(x, hi - lo, left=lo, height=0.5, alpha=0.4, color="steelblue",
            label="90% credible interval")
    ax.scatter(obs, x, marker="o", color="crimson", s=50, zorder=5, label="Observed rate")
    ax.scatter(post, x, marker="D", color="navy", s=40, zorder=6, label="Posterior mean")
    ax.axvline(grand_mean_rate, color="grey", linestyle=":", alpha=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(districts_plot, fontsize=8)
    ax.set_xlabel("Claim frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.suptitle("90% posterior credible intervals\n(navy diamond = posterior mean, red dot = observed)", fontsize=11)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Shows credible intervals for the 20 densest and 20 thinnest districts. This is the chart that answers the pricing committee question "how confident are we in this rate?"

**Run this cell.**

**What you should see:**
- Left panel (dense districts): Narrow bars. The posterior mean (navy diamond) sits close to the observed rate (red dot) — high credibility, trusting own experience.
- Right panel (thin districts): Wide bars spanning much of the plausible rate range. The posterior mean sits noticeably closer to the grand mean (grey dotted line) than the observed rate — strong shrinkage toward the portfolio mean.

This is the honest picture. A thin district's rate might genuinely be anywhere in that wide interval. The pricing decision is: set the rate at the posterior mean, acknowledge the uncertainty, and monitor as experience develops.