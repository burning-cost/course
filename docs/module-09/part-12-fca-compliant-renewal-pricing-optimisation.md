## Part 12: FCA-compliant renewal pricing optimisation

The demand curve tells you the portfolio-level optimum. The `RenewalPricingOptimiser` applies the optimisation at the individual policy level, subject to the PS21/5 ENBP constraint.

### How the per-policy optimisation works

For each customer, the optimiser:

1. Takes their estimated CATE (individual price sensitivity)
2. Uses their technical premium as the cost floor
3. Uses their ENBP as the price ceiling (PS21/5)
4. Finds the price that maximises expected profit, given the linear demand approximation: P(renew | new_price) = P0 + CATE x delta_log_price

This is a small single-variable optimisation problem for each customer, solved using a 50-point grid search over the feasible range. For a portfolio of 50,000 customers, this takes a few seconds.

### Running the optimiser

```python
%md
## Part 12: Per-policy optimisation
```

```python
opt = RenewalPricingOptimiser(
    est_renewal,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,  # do not price below technical premium
)

priced_df = opt.optimise(df_renewals, objective="profit")

print("Optimisation results:")
print(f"Policies optimised: {len(priced_df):,}")
print(f"Mean optimal price: £{priced_df['optimal_price'].mean():.2f}")
print(f"Mean last premium:  £{priced_df['last_premium'].mean():.2f}")
print(f"Mean price change:  {((priced_df['optimal_price'] / priced_df['last_premium']).mean() - 1) * 100:.1f}%")
print(f"Mean ENBP headroom: £{priced_df['enbp_headroom'].mean():.2f}")
print(f"Predicted renewal rate: {priced_df['predicted_renewal_prob'].mean():.3f}")
print(f"Expected profit/policy: £{priced_df['expected_profit'].mean():.2f}")
```

The `enbp_headroom` column shows how far below the ENBP ceiling the optimal price sits. Negative headroom would mean a breach - the optimiser should never produce this. A positive headroom of £0 means the ENBP constraint is binding: the profit-maximising price is above ENBP, but we cannot charge it.

```python
# How often is the ENBP constraint binding?
binding = (priced_df["enbp_headroom"] < 1.0).sum()
total = len(priced_df)
print(f"ENBP constraint binding: {binding:,} of {total:,} policies ({100*binding/total:.1f}%)")
```

A high proportion of binding ENBP constraints tells you that the profitable action would be to charge more, but the regulation prevents it. This is the quantified "regulatory cost" of PS21/5 that we mentioned in Part 1.

### Understanding the optimal price distribution

```python
# Optimal price relative to ENBP
priced_df_pd = priced_df.to_pandas()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: distribution of optimal prices relative to ENBP
ratio = priced_df["optimal_price"] / priced_df["enbp"]
axes[0].hist(ratio.to_numpy(), bins=40, color="#1f77b4", alpha=0.7)
axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="ENBP ceiling")
axes[0].set_xlabel("Optimal price / ENBP")
axes[0].set_ylabel("Count")
axes[0].set_title("Optimal price as fraction of ENBP")
axes[0].legend()

# Right: expected profit by NCD band
ncd_profit = (
    priced_df.group_by("ncd_years")
    .agg(pl.col("expected_profit").mean().alias("mean_profit"))
    .sort("ncd_years")
)
axes[1].bar(ncd_profit["ncd_years"].to_numpy(), ncd_profit["mean_profit"].to_numpy(),
            color="#2ca02c", alpha=0.8)
axes[1].set_xlabel("NCD years")
axes[1].set_ylabel("Expected profit per policy (£)")
axes[1].set_title("Expected profit by NCD band")

plt.tight_layout()
plt.savefig("/tmp/optimisation_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

The left chart should show a cluster of policies priced at the ENBP ceiling (ratio near 1.0) and a spread of policies priced below. The right chart shows that higher-NCD customers tend to be more profitable - they are inelastic and can be priced closer to the ceiling.