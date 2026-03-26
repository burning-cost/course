## Part 11: Working with per-policy multipliers

The solver returns a multiplier array of shape `(N,)`. Every policy has its own optimal price. This is more expressive than a factor-table approach but requires additional steps to communicate to the rating engine and to the pricing committee.

### Per-policy result DataFrame

The full per-policy output is in `result.summary_df`:

```python
summary = result.summary_df

print("Per-policy result (sample):")
print(summary.head(10))
print()

# Policies at the ENBP cap
enbp_at_cap = summary.filter(pl.col("enbp_binding"))
print(f"Policies at ENBP cap: {len(enbp_at_cap):,} of {renewal_flag.sum():,} renewals")

# Rate change distribution
print("\nRate change distribution:")
print(f"  Mean:    {summary['rate_change_pct'].mean():+.1f}%")
print(f"  Median:  {summary['rate_change_pct'].median():+.1f}%")
print(f"  p10:     {summary['rate_change_pct'].quantile(0.10):+.1f}%")
print(f"  p90:     {summary['rate_change_pct'].quantile(0.90):+.1f}%")
```

Columns in `summary_df`:
- `policy_idx`: integer index (0 to N-1)
- `multiplier`: optimal price multiplier (`new_premium / technical_price`)
- `new_premium`: optimal final premium
- `expected_demand`: expected demand at the optimal price
- `contribution`: per-policy profit contribution `(new_premium - expected_loss_cost) * expected_demand`
- `enbp_binding`: True if this policy's multiplier hits the ENBP upper bound
- `rate_change_pct`: year-on-year rate change as a percentage

### Summarising to segments for implementation

The rating engine typically implements rates as factor tables, not per-policy multipliers. To bridge this, group the per-policy multipliers into the segments your rating system knows about and compute the mean multiplier per segment.

```python
# Build analysis DataFrame with segment columns
df_with_results = df.with_columns([
    pl.Series("optimal_multiplier", result.multipliers.tolist()),
    pl.Series("new_premium", result.new_premiums.tolist()),
    pl.Series("rate_change_pct", result.summary_df["rate_change_pct"].to_list()),
])

# Segment-level mean multipliers (for rating engine communication)
segment_summary = (
    df_with_results
    .group_by(["channel"])
    .agg([
        pl.len().alias("n_policies"),
        pl.col("optimal_multiplier").mean().alias("mean_multiplier"),
        pl.col("rate_change_pct").mean().alias("mean_rate_change_pct"),
        pl.col("new_premium").mean().alias("mean_new_premium"),
    ])
    .sort("channel")
)
print("Segment-level summary:")
print(segment_summary)
```

The mean multiplier per segment is the number to communicate to the rating engine: "for PCW renewals, apply a 1.045x multiplier to current premiums." This rounds the continuous per-policy solution back to a discrete segment action.

### The "shape" vs "scale" distinction

The per-policy optimisation produces a continuous multiplier surface across the portfolio. When you aggregate to segments, you are choosing to implement the solution approximately. There are two ways to think about this:

**Shape-preserving implementation.** Apply the segment mean multiplier uniformly to all policies in that segment. This preserves the relative ordering of premiums within the segment (same percentage change for all).

**Full per-policy implementation.** Load the policy-level multipliers directly into the rating engine. This is the exact solution but requires the rating engine to support per-policy overrides — most modern systems do. It is more accurate but harder to explain to compliance and the pricing committee.

For the pricing committee, the shape-preserving approach is easier to communicate. For the rating engine, per-policy is more accurate. In practice, most teams implement a hybrid: per-policy multipliers for the top and bottom of the distribution (where the differentiation matters most), and segment averages elsewhere.

### What the pricing committee needs to see

The key outputs for the pricing committee pack:

1. The efficient frontier plot (from Part 10): shows the profit-retention trade-off
2. The distribution of rate changes: mean, median, 10th–90th percentile, and the number of policies seeing >10% or <-10% change
3. The ENBP compliance statement: zero violations per-policy
4. The shadow prices: which constraints are binding and their marginal cost
5. The convergence flag: `result.converged = True`

Shadow prices are in `result.shadow_prices`. A positive value means the constraint is binding:

```python
print("Binding constraints:")
for name, sp in result.shadow_prices.items():
    binding = "BINDING" if abs(sp) > 1e-6 else "slack"
    print(f"  {name:<20}: shadow price = {sp:+.4f}  [{binding}]")
```

The shadow price on `lr_max` has the interpretation: if you relax the LR cap by 0.001 (from 72% to 72.1%), expected profit increases by approximately `shadow_prices['lr_max'] * 0.001`. This is the quantification the pricing committee needs to decide whether a different LR target is worth pursuing.
