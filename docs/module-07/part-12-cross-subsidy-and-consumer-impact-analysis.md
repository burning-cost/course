## Part 12: Cross-subsidy and consumer impact analysis

### Per-policy optimisation produces heterogeneous rate changes

Unlike a uniform factor-table adjustment where every customer sees the same percentage change, a per-policy optimisation produces different rate changes for different customers. The distribution reflects the demand model: high-elasticity customers (PCW, newer) receive smaller increases, low-elasticity customers (direct, tenured) receive larger increases.

This is correct and commercially desirable — but it requires a different consumer impact analysis than a uniform rate action.

### Computing the individual premium distribution

```python
import polars as pl
import numpy as np

df_analysis = df.with_columns([
    pl.Series("new_premium",      result.new_premiums.tolist()),
    pl.Series("multiplier",       result.multipliers.tolist()),
    pl.Series("rate_change_pct",  result.summary_df["rate_change_pct"].to_list()),
]).with_columns([
    (pl.col("new_premium") - pl.col("current_premium")).alias("abs_change_gbp"),
])

print("Portfolio premium impact:")
print(f"  Mean change:              {df_analysis['rate_change_pct'].mean():+.1f}%")
print(f"  Median change:            {df_analysis['rate_change_pct'].median():+.1f}%")
print(f"  10th percentile:          {df_analysis['rate_change_pct'].quantile(0.10):+.1f}%")
print(f"  90th percentile:          {df_analysis['rate_change_pct'].quantile(0.90):+.1f}%")
print(f"  Mean abs increase:        £{df_analysis['abs_change_gbp'].mean():.2f}")
print(f"  Median abs increase:      £{df_analysis['abs_change_gbp'].median():.2f}")
print()
print(f"  Policies with >+10%:      {(df_analysis['rate_change_pct'] > 10).sum():,}")
print(f"  Policies with >+20%:      {(df_analysis['rate_change_pct'] > 20).sum():,}")
print(f"  Policies with decrease:   {(df_analysis['rate_change_pct'] < 0).sum():,}")

# Cross-subsidy by channel
channel_summary = (
    df_analysis
    .group_by("channel")
    .agg([
        pl.len().alias("n"),
        pl.col("rate_change_pct").mean().alias("mean_pct_change"),
        pl.col("abs_change_gbp").mean().alias("mean_abs_change_gbp"),
        pl.col("current_premium").mean().alias("mean_current_prem"),
        pl.col("new_premium").mean().alias("mean_new_prem"),
    ])
    .sort("channel")
)
print("\nPremium impact by channel:")
print(channel_summary)
```

**What you should see:**

```text
Portfolio premium impact:
  Mean change:              +4.8%
  Median change:            +4.2%
  10th percentile:          +0.3%
  90th percentile:          +9.8%
  Mean abs increase:        £24.31
  Median abs increase:      £21.67

  Policies with >+10%:      423
  Policies with >+20%:      12
  Policies with decrease:   87

Premium impact by channel:
channel  n      mean_pct_change  mean_abs_change_gbp  mean_current_prem  mean_new_prem
PCW      2950   +3.1%            £17.82               £482.12            £497.06
direct   2050   +7.3%            £33.65               £476.88            £511.67
```

PCW customers see smaller percentage increases (they are more elastic — the optimiser is careful not to push them into lapsing). Direct customers see larger increases (they are less elastic — the optimiser extracts more value here).

### What Consumer Duty requires

Consumer Duty (PS22/9) requires you to confirm that no customer segment is being treated unfairly. For a per-policy optimisation, the relevant question is: does the differential rate change create a systematically unfair outcome for any segment?

The correct framing for the Consumer Duty evidence file:

"Rate changes vary by customer because the optimiser differentiates based on price sensitivity (elasticity). Customers who are more price-sensitive receive smaller increases because higher increases for them would cause a disproportionate increase in lapses, which would harm those customers. Less price-sensitive customers, who have demonstrated a stronger preference to stay regardless of price, receive larger increases. This differentiation is not by demographic characteristic — it is by measured price behaviour. No customer group receives a larger increase because of their protected characteristic (age, disability, etc.). We have verified that the rate change distribution is not systematically correlated with [protected characteristics] at the 5% significance level."

If you do find a correlation with protected characteristics, that is a regulatory issue to escalate before proceeding with the rate action. The `df_analysis` DataFrame gives you everything you need to run this check.

### Rate change distribution chart

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: overall rate change distribution
rc = df_analysis["rate_change_pct"].to_numpy()
axes[0].hist(rc, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
axes[0].axvline(rc.mean(), color="firebrick", linestyle="--",
                label=f"Mean: {rc.mean():+.1f}%")
axes[0].axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
axes[0].set_xlabel("Rate change (%)")
axes[0].set_ylabel("Number of policies")
axes[0].set_title("Distribution of individual rate changes")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: by channel
for channel_name, colour in [("PCW", "steelblue"), ("direct", "darkorange")]:
    rc_ch = (
        df_analysis
        .filter(pl.col("channel") == channel_name)
        ["rate_change_pct"]
        .to_numpy()
    )
    axes[1].hist(rc_ch, bins=40, alpha=0.6, color=colour,
                 edgecolor="white", linewidth=0.3, label=channel_name)

axes[1].axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
axes[1].set_xlabel("Rate change (%)")
axes[1].set_ylabel("Number of policies")
axes[1].set_title("Rate change distribution by channel")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Motor renewal book — Q2 2026 rate action: customer impact", fontsize=13)
plt.tight_layout()
plt.show()
```
