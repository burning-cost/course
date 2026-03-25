## Part 9: Per-customer elasticity estimates

Beyond segment averages, `CausalForestDML` produces a per-customer CATE estimate: the expected change in that individual's renewal probability for a unit change in log price. This is the input to the per-policy optimisation in Part 12.

### Extracting CATEs

```python
%md
## Part 9: Per-customer CATE estimates
```

```python
cate_values = est.cate(df)

print("CATE distribution across 50,000 customers:")
print(f"  Mean:  {cate_values.mean():.3f}  (should be close to ATE: {ate:.3f})")
print(f"  Std:   {cate_values.std():.3f}")
print(f"  Min:   {cate_values.min():.3f}")
print(f"  Max:   {cate_values.max():.3f}")
print(f"  10th:  {np.percentile(cate_values, 10):.3f}")
print(f"  90th:  {np.percentile(cate_values, 90):.3f}")
```

The spread of CATEs across customers reflects the genuine heterogeneity in price sensitivity. A standard deviation of 0.8–1.2 on the CATE distribution means the most elastic customers are roughly 3–5× more sensitive to price than the least elastic.

### Per-customer confidence intervals

```python
lb_vals, ub_vals = est.cate_interval(df, alpha=0.05)

# Add to DataFrame for analysis
df_with_cate = df.with_columns([
    pl.Series("cate", cate_values),
    pl.Series("cate_lower", lb_vals),
    pl.Series("cate_upper", ub_vals),
])

# What fraction of customers have a CATE CI that excludes zero?
n_significant = int(((df_with_cate["cate_upper"] < 0)).sum())
print(f"Customers with significantly negative elasticity: {n_significant:,} "
      f"({100 * n_significant / len(df):.1f}%)")
```

A customer with a CATE CI that does not include zero has a statistically distinguishable elasticity from zero. For pricing decisions, you may want to treat customers with wide, zero-straddling CIs conservatively — using the ATE rather than the per-customer estimate.

### Validating the CATE recovery

In synthetic data we can compare each customer's CATE estimate against their true elasticity:

```python
cate_bias = cate_values - df["true_elasticity"].to_numpy()

print("CATE recovery validation:")
print(f"  Mean bias:    {cate_bias.mean():.4f}  (should be near zero)")
print(f"  RMSE:         {np.sqrt(np.mean(cate_bias**2)):.4f}")
print(f"  Correlation (estimated vs. true): "
      f"{np.corrcoef(cate_values, df['true_elasticity'].to_numpy())[0,1]:.4f}")
```

A correlation above 0.7 between estimated and true CATEs indicates the forest is picking up meaningful individual-level heterogeneity, not just noise. On 50,000 observations, expect correlation around 0.75–0.85.

### Distribution of CATEs by segment

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, ["ncd_years", "channel", "age_band"]):
    for group, group_df in df_with_cate.group_by(col):
        label = str(group[0]) if isinstance(group, tuple) else str(group)
        ax.hist(group_df["cate"].to_numpy(), bins=40, alpha=0.5, label=label, density=True)
    ax.set_xlabel("CATE (semi-elasticity)")
    ax.set_title(f"CATE distribution by {col}")
    ax.legend(fontsize=7)
    ax.axvline(ate, color="black", linestyle="--", linewidth=1, label="ATE")

plt.tight_layout()
plt.savefig("/tmp/cate_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
```

Each subplot should show clearly separated distributions by group. NCD=0 CATEs should sit well to the left of NCD=5. PCW CATEs should sit left of direct. Young driver CATEs should sit left of 65+. If the distributions overlap completely, the estimator has not recovered meaningful heterogeneity — check that the confounders list is correctly specified and that price variation is sufficient.
