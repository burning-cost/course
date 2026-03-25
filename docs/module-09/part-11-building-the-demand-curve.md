## Part 11: The portfolio demand curve

The per-customer CATEs are the right tool for individual-policy optimisation. Before going there, build the portfolio demand curve — a sweep of price changes and their expected aggregate outcomes. This is the management information view: the chart that goes to the commercial director.

### Computing the demand curve

```python
%md
## Part 11: Portfolio demand curve
```

```python
# Sweep price changes from -25% to +25% in 50 steps
demand_df = demand_curve(
    est,
    df,
    price_range=(-0.25, 0.25, 50),
)

print("Demand curve (selected price points):")
print(
    demand_df
    .filter(pl.col("pct_price_change").is_between(-0.15, 0.15))
    .with_columns([
        (pl.col("pct_price_change") * 100).round(1).alias("price_change_%"),
        (pl.col("predicted_renewal_rate") * 100).round(2).alias("renewal_rate_%"),
        pl.col("predicted_profit").round(2),
    ])
    .select(["price_change_%", "renewal_rate_%", "predicted_profit"])
)
```

### Finding the portfolio-level optimum

```python
max_profit_row = demand_df.sort("predicted_profit", descending=True).row(0, named=True)

print(f"Profit-maximising price change: {max_profit_row['pct_price_change'] * 100:.1f}%")
print(f"Renewal rate at optimum:        {max_profit_row['predicted_renewal_rate'] * 100:.1f}%")
print(f"Expected profit per policy:     £{max_profit_row['predicted_profit']:.2f}")
```

### Plotting the demand curve

```python
fig_demand = plot_demand_curve(demand_df, show_profit=True)
fig_demand.savefig("/tmp/demand_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

The dual-axis chart shows:

- **Left axis (blue)**: renewal rate falling as price change increases — the demand curve itself
- **Right axis (red)**: expected profit per policy — a hump shape. Rising at first (higher margin more than offsets volume loss), then falling (too many lapses)
- **Red dot**: the profit-maximising price change

This is the commercial director's chart. It frames the decision: "the data says an X% price increase maximises expected profit, at the cost of Y% lower renewal rate. If we accept a lower renewal rate target, we can take more rate."

### Why the portfolio optimum is not what you implement

The portfolio demand curve uses the average CATE. It tells you what happens if you apply the same price change to everyone. It does not account for the fact that an NCD-0 PCW customer has 3× the price sensitivity of an NCD-5 direct customer, or that some customers are already at the ENBP ceiling and cannot be priced higher.

The per-policy optimiser in Part 12 uses the individual-level CATEs and the per-policy ENBP constraint to find the best price for each customer separately. The portfolio demand curve is a benchmark and an MI tool. The per-policy optimiser is the operational output.
