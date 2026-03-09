## Part 12: Plotting for committee presentation

The library has a built-in plotting function. In a new cell, type this and run it (Shift+Enter):

```python
sr.plot_relativities(
    features=["area", "has_convictions"],
    show_ci=True,
    figsize=(12, 5),
)
display(plt.gcf())
```

You will see two bar charts side by side: area bands and conviction flag. Each bar shows the relativity, with whiskers for the 95% confidence interval. The base level sits at 1.0.

On Databricks, you must call `display(plt.gcf())` after `plot_relativities()` to render the chart in the notebook. Without it, the chart appears in some Databricks environments but not others.

Now produce a more polished age comparison chart manually. In a new cell, type this and run it (Shift+Enter):

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: GBM LOESS curve vs GLM linear prediction
age_range = np.arange(17, 86)
glm_age_pred = np.exp(glm_age_coef * (age_range - base_age))

axes[0].plot(age_curve["feature_value"], age_curve["relativity"],
             color="steelblue", lw=2, label="GBM (LOESS)")
axes[0].plot(age_range, glm_age_pred,
             color="firebrick", lw=2, linestyle="--", label="GLM (linear)")
axes[0].axhline(1.0, color="grey", linestyle=":", alpha=0.5)
axes[0].set_xlabel("Driver age")
axes[0].set_ylabel("Relativity")
axes[0].set_title("Driver age: GBM vs GLM")
axes[0].legend()
axes[0].set_ylim(0.5, 2.5)

# Right: GBM banded relativities
band_rels_sorted = band_rels.sort("age_band")
band_labels = band_rels_sorted["age_band"].to_list()
band_values = band_rels_sorted["relativity"].to_list()
x_pos = range(len(band_labels))

axes[1].bar(x_pos, band_values, color="steelblue", alpha=0.8)
axes[1].axhline(1.0, color="grey", linestyle="--", alpha=0.5)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(band_labels, rotation=30, ha="right")
axes[1].set_ylabel("Relativity")
axes[1].set_title("Driver age: banded relativities (base: 30-39)")

plt.tight_layout()
display(fig)
```

You will see two charts. The left chart shows the GBM's smooth LOESS curve alongside the GLM's linear prediction. The U-shape of the GBM curve is the key result. The right chart shows the banded version ready for a factor table.