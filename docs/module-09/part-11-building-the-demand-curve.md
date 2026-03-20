## Part 11: Building the demand curve

The demand curve translates the elasticity model into a concrete view of the volume-profit trade-off across a range of price changes. It is a management information tool, not a per-customer optimiser.

```python
%md
## Part 11: Portfolio demand curve
```

```python
from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve

# Sweep price changes from -25% to +25%
demand_df = demand_curve(
    est_renewal,
    df_renewals,
    price_range=(-0.25, 0.25, 50),
)

print("Demand curve (selected rows):")
print(demand_df.select([
    "pct_price_change",
    "predicted_renewal_rate",
    "predicted_profit",
]).filter(
    pl.col("pct_price_change").is_between(-0.15, 0.15)
).with_columns(
    (pl.col("pct_price_change") * 100).round(1).alias("pct_change_%"),
    (pl.col("predicted_renewal_rate") * 100).round(2).alias("renewal_rate_%"),
    pl.col("predicted_profit").round(2),
).select(["pct_change_%", "renewal_rate_%", "predicted_profit"])
)
```

You should see that:
- At -25% price change: very high renewal rate, low profit per policy
- At 0% price change: current market rates
- At +25% price change: renewal rate has fallen, but higher margin on those who stay

The profit-maximising price change is where expected profit per policy is maximised - not necessarily at 0% change. The demand curve makes this visible.

```python
# Find the profit-maximising price change
max_profit_row = demand_df.sort("predicted_profit", descending=True).row(0, named=True)
print(f"\nProfit-maximising price change: {max_profit_row['pct_price_change']*100:.1f}%")
print(f"Expected renewal rate at that point: {max_profit_row['predicted_renewal_rate']*100:.1f}%")
print(f"Expected profit per policy: £{max_profit_row['predicted_profit']:.2f}")
```

Now plot it:

```python
fig_demand = plot_demand_curve(demand_df, show_profit=True)
fig_demand.savefig("/tmp/demand_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

The dual-axis plot shows the renewal rate (left axis, blue) falling as price increases, and the expected profit per policy (right axis, red) showing a hump shape - rising at first as the higher margin more than offsets the volume loss, then falling as too many customers lapse. The red dot marks the profit peak.

This chart is what you show the commercial director. It translates the technical DML output into a concrete business decision: "the data suggests that a portfolio-wide price increase of X% would maximise expected profit, at the cost of Y% lower renewal rate."