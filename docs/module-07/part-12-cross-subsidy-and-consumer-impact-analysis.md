## Part 12: Cross-subsidy and consumer impact analysis

### The uniform percentage result

Before running any cross-subsidy code, state clearly what a uniform factor action implies: every customer in the portfolio receives approximately the same percentage premium increase.

This follows directly from the structure of the optimiser. The decision variables are multiplicative factors applied uniformly to all levels of each factor table. If all five factors increase by 3.7%, the combined multiplier for any policy is:

```
1.0368 x 1.0368 x 1.0368 x 1.0368 x 1.0 = 1.0368^4 = 1.158
```

Wait — that would be 15.8%, not 3.7%. The key insight is that each factor affects the premium multiplicatively, but the optimiser adjusts the factor table itself, not the combined premium. A policy in the low-age band (relativity 1.00) sees its age factor go from 1.00 to 1.037. A policy in the high-age band (relativity 2.00) sees its age factor go from 2.00 to 2.074. Both see a 3.7% change in the age factor. The ratio of old to new is the same regardless of which level of the table you are on.

The combined adjustment for each policy is the product of all factor adjustments: `1.037 x 1.036 x 1.036 x 1.036 x 1.000 = 1.151`, approximately 15.1%. This is the same for all policies.

### Computing the individual premium distribution

```python
# Compute the combined adjustment multiplier for each policy
combined_adj = 1.0
for fname in FACTOR_NAMES:
    m = factor_adj.get(fname, 1.0)
    combined_adj *= m

# Apply to the Polars DataFrame
df_analysis = df.with_columns([
    (pl.col("current_premium") * combined_adj).alias("new_premium"),
    ((combined_adj - 1) * 100 * pl.lit(1.0)).alias("pct_change"),
    (pl.col("current_premium") * (combined_adj - 1)).alias("abs_change_gbp"),
])

print("Portfolio premium impact:")
print(f"  Combined adjustment: {combined_adj:.4f} = {(combined_adj-1)*100:+.1f}%")
print(f"  Mean premium increase:   £{df_analysis['abs_change_gbp'].mean():.2f}")
print(f"  Median premium increase: £{df_analysis['abs_change_gbp'].median():.2f}")
print()

# Cross-subsidy analysis: by age band
# Percentage change is uniform; absolute change varies with the premium level
print("Premium impact by age relativity band (absolute change, not percentage):")
print("Note: percentage change is identical for all customers (~15.1%).")
print("The variation in absolute impact is driven by the current premium level,")
print("not by the rate action itself.\n")

age_bands = df_analysis.with_columns(
    pl.col("f_age").cast(pl.Utf8).alias("age_band")
).group_by("age_band").agg([
    pl.len().alias("n_policies"),
    pl.col("current_premium").mean().alias("mean_current_premium"),
    pl.col("new_premium").mean().alias("mean_new_premium"),
    pl.col("abs_change_gbp").mean().alias("mean_abs_increase_gbp"),
    pl.col("pct_change").mean().alias("mean_pct_change"),
]).sort("age_band")

print(age_bands)
```

**What you should see:**

```
Portfolio premium impact:
  Combined adjustment: 1.1512 = +15.1%
  Mean premium increase:   £72.43
  Median premium increase: £68.21

Premium impact by age relativity band (absolute change, not percentage):
Note: percentage change is identical for all customers (~15.1%).
The variation in absolute impact is driven by the current premium level,
not by the rate action itself.

age_band   n_policies  mean_current_premium  mean_new_premium  mean_abs_increase_gbp  mean_pct_change
0.8               750               £242.11           £278.66               £36.55            15.1%
1.0              1500               £302.64           £348.35               £45.71            15.1%
1.2              1500               £363.17           £418.04               £54.87            15.1%
1.5               750               £453.96           £522.45               £68.49            15.1%
2.0               500               £605.28           £696.55               £91.27            15.1%
```

The percentage change is identical across all age bands. The absolute increase is larger for young drivers (2.0x band) than middle-aged drivers (1.0x band) — not because the rate action targets them disproportionately, but because they start from a higher premium base.

### What Consumer Duty requires

Consumer Duty (PS 22/9) requires you to confirm that no customer segment is being treated unfairly. For a uniform rate action, the relevant question is: does the absolute premium increase create affordability concerns for any segment?

Young drivers paying £605 per year will see their premium rise to £697 — a £91 increase. This is a legitimate affordability concern to document, even though it results from the same percentage change applied to a higher base premium. The FCA expects you to have reviewed this explicitly, not to have noted only the percentage.

The correct framing for the Consumer Duty evidence file is: "The rate action applies a uniform percentage increase across all factor levels. The absolute premium increase is higher for customers with high-risk profiles (younger drivers, high-vehicle-group) because their base premium is higher. No differential treatment has been applied to any segment; the variation in absolute impact is a function of the existing tariff structure."