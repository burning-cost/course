## Part 12: Per-policy profit optimisation

The `RenewalPricingOptimiser` applies the elasticity estimates to find the profit-maximising renewal price for each policy individually, subject to the PS21/5 ENBP constraint.

### How the per-policy optimisation works

For each customer, the optimiser:

1. Takes their estimated CATE (individual price sensitivity from the fitted forest)
2. Uses their `tech_prem` as the cost floor — the price cannot go below this
3. Uses their `enbp` as the hard ceiling — the FCA constraint
4. Sweeps 50 candidate prices across the feasible range [tech_prem, enbp]
5. Evaluates expected profit at each: (price − tech_prem) × P(renew | price)
6. Returns the price with the highest expected profit

The demand model used is linear in log price: P(renew | new_price) = P₀ + CATE × Δ(log price), where P₀ is the observed renewal indicator smoothed with the portfolio rate and Δ(log price) is the change from current offer price to the candidate price. For price changes in [−20%, +20%], the linear approximation is accurate to within 1–2 percentage points of the logistic model.

### Running the optimiser

```python
%md
## Part 12: Per-policy optimisation
```

```python
opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,   # do not price below technical premium
)

priced_df = opt.optimise(df, objective="profit")

print(f"Policies optimised:      {len(priced_df):,}")
print(f"Mean optimal price:      £{priced_df['optimal_price'].mean():.2f}")
print(f"Mean last premium:       £{priced_df['last_premium'].mean():.2f}")
pct_change = ((priced_df["optimal_price"] / priced_df["last_premium"]).mean() - 1) * 100
print(f"Mean price change:       {pct_change:.1f}%")
print(f"Mean ENBP headroom:      £{priced_df['enbp_headroom'].mean():.2f}")
print(f"Predicted renewal rate:  {priced_df['predicted_renewal_prob'].mean():.3f}")
print(f"Expected profit/policy:  £{priced_df['expected_profit'].mean():.2f}")
```

The `enbp_headroom` column is the gap between the optimal price and the ENBP. Positive headroom means the ENBP is not binding — the optimal price is below the ceiling. Headroom near zero means the profit-maximising price is right at the ENBP ceiling.

### How often does the ENBP constraint bind?

```python
binding = (priced_df["enbp_headroom"] < 1.0).sum()
total = len(priced_df)
print(f"ENBP constraint binding: {binding:,} of {total:,} policies "
      f"({100 * binding / total:.1f}%)")
```

A high proportion of binding constraints tells you the profitable action would be to charge more, but the regulation prevents it. This is the quantified regulatory cost of PS21/5. On the synthetic data the proportion should be moderate — in real post-PS21/5 books, particularly for high-tenure segments, this can exceed 60%.

### Visualising the results

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: optimal price as fraction of ENBP ceiling
ratio = (priced_df["optimal_price"] / priced_df["enbp"]).to_numpy()
axes[0].hist(ratio, bins=40, color="#1f77b4", alpha=0.8)
axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="ENBP ceiling")
axes[0].set_xlabel("Optimal price / ENBP")
axes[0].set_ylabel("Count")
axes[0].set_title("Optimal price relative to ENBP ceiling")
axes[0].legend()

# Right: expected profit by NCD band
ncd_profit = (
    priced_df
    .group_by("ncd_years")
    .agg(pl.col("expected_profit").mean().alias("mean_profit"))
    .sort("ncd_years")
)
axes[1].bar(
    ncd_profit["ncd_years"].to_numpy(),
    ncd_profit["mean_profit"].to_numpy(),
    color="#2ca02c", alpha=0.8,
)
axes[1].set_xlabel("NCD years")
axes[1].set_ylabel("Expected profit per policy (£)")
axes[1].set_title("Expected profit by NCD band")

plt.tight_layout()
plt.savefig("/tmp/optimisation_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

The left chart shows a cluster of policies priced at or near the ENBP ceiling (ratio near 1.0) — these are the inelastic customers for whom the profit-maximising price would have been above ENBP if the constraint were not there. The right chart shows that higher-NCD customers generate more expected profit per policy: they are inelastic (less volume response to price) and can be priced nearer the ceiling.
