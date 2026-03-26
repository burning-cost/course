## Part 10: The efficient frontier

A single solve gives you one point: the optimal multipliers at a specific retention floor. The efficient frontier gives you the full curve: all achievable (retention, profit) combinations as the retention floor varies.

This is the tool for the pricing committee conversation. Instead of asking "should we use a 85% or 87% retention floor?", you ask "here is the frontier; which point on it do we want to operate at?"

### Tracing the frontier

```python
from insurance_optimise import EfficientFrontier

frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="volume_retention",
    sweep_range=(0.80, 0.97),
    n_points=15,
    n_jobs=1,
)

frontier_result = frontier.run()

print("Efficient frontier:")
print(frontier_result.data)
```

`EfficientFrontier` takes:
- `optimiser`: a configured `PortfolioOptimiser` (the base config is used for all shared constraints)
- `sweep_param`: which constraint to vary. Supported: `"volume_retention"`, `"gwp_min"`, `"lr_max"`
- `sweep_range`: `(min_value, max_value)` for the sweep
- `n_points`: number of points on the frontier (default 15)

Each point is an independent optimisation problem solved from scratch. The base constraints (`lr_max`, `max_rate_change`, ENBP) are fixed; the swept constraint changes at each point.

**What you should see** (abbreviated):

```text
shape: (15, 6)
┌─────────┬───────────┬─────────────┬──────────────┬──────────┬───────────┐
│ epsilon ┆ converged ┆ profit      ┆ gwp          ┆ loss_rat ┆ retention │
│ ---     ┆ ---       ┆ ---         ┆ ---          ┆ ---      ┆ ---       │
│ f64     ┆ bool      ┆ f64         ┆ f64          ┆ f64      ┆ f64       │
╞═════════╪═══════════╪═════════════╪══════════════╪══════════╪═══════════╡
│ 0.800   ┆ true      ┆ 1_523_456.0 ┆ 7_112_345.0  ┆ 0.7200   ┆ 0.8034   │
│ 0.811   ┆ true      ┆ 1_498_123.0 ┆ 7_098_234.0  ┆ 0.7200   ┆ 0.8110   │
│ …       ┆ …         ┆ …           ┆ …            ┆ …        ┆ …        │
│ 0.970   ┆ false     ┆ nan         ┆ nan          ┆ nan      ┆ nan       │
└─────────┴───────────┴─────────────┴──────────────┴──────────┴───────────┘
```

The `epsilon` column is the retention floor at that point. The frontier may become infeasible at tight retention floors — these appear as `converged=False` with `nan` metrics.

### Reading the frontier DataFrame

`frontier_result.data` is a Polars DataFrame with columns:
- `epsilon`: the swept constraint value (retention floor at this point)
- `converged`: True if this point converged to a valid solution
- `profit`: expected profit at the optimum
- `gwp`: expected GWP
- `loss_ratio`: expected LR
- `retention`: expected renewal retention

Filter to converged points only with `frontier_result.pareto_data()`.

### Understanding the frontier shape

At loose retention floors (0.80), the optimiser has lots of freedom to take rate. Profit is highest and LR hits the target with room to spare. As the retention floor tightens, the optimiser must sacrifice profit to preserve volume — it cuts rates on high-elasticity customers to keep them from lapsing. At some point, the LR and retention constraints cannot both be satisfied simultaneously, and the problem becomes infeasible.

The **knee** of the frontier is where the profit starts falling steeply relative to the retention gain. This is the natural operating point: beyond the knee, each additional percentage point of retention you protect costs disproportionately more profit.

### Computing the knee

```python
pareto = frontier_result.pareto_data().to_pandas().sort_values("epsilon")

# Profit drop per percentage point of retention gained
pareto["profit_drop_per_ret_pp"] = (
    -pareto["profit"].diff() / pareto["retention"].diff() / 100
)

print("Profit cost of each retention percentage point:")
print(pareto[["epsilon", "retention", "profit", "profit_drop_per_ret_pp"]].to_string(index=False))

# Knee: where the cost first exceeds 2x the median cost
median_cost = pareto["profit_drop_per_ret_pp"].median()
knee_rows = pareto[pareto["profit_drop_per_ret_pp"] >= 2 * median_cost]
if not knee_rows.empty:
    knee = knee_rows.iloc[0]
    print(f"\nKnee of the efficient frontier:")
    print(f"  Retention floor: {knee['epsilon']:.3f}")
    print(f"  Expected profit: £{knee['profit']:,.0f}")
    print(f"  Expected LR:     {knee['loss_ratio']:.4f}")
```

### Plotting the frontier for the pricing committee

```python
import matplotlib.pyplot as plt

pareto_pd = frontier_result.pareto_data().to_pandas().sort_values("retention")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: profit vs retention
ax1.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["profit"] / 1_000,
    "o-", color="steelblue", linewidth=2, markersize=5,
)
if not knee_rows.empty:
    ax1.scatter(
        [knee["retention"] * 100],
        [knee["profit"] / 1_000],
        color="firebrick", s=100, zorder=5, label="Knee",
    )
ax1.axvline(RETENTION_FLOOR * 100, linestyle="--", color="grey",
            alpha=0.5, label=f"Retention floor ({RETENTION_FLOOR:.0%})")
ax1.set_xlabel("Expected retention (%)", fontsize=11)
ax1.set_ylabel("Expected profit (£k)", fontsize=11)
ax1.set_title("Efficient frontier: profit vs retention", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: loss ratio vs retention
ax2.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["loss_ratio"] * 100,
    "o-", color="darkorange", linewidth=2, markersize=5,
)
ax2.axhline(LR_TARGET * 100, linestyle="--", color="firebrick",
            alpha=0.6, label=f"LR target ({LR_TARGET:.0%})")
ax2.set_xlabel("Expected retention (%)", fontsize=11)
ax2.set_ylabel("Expected loss ratio (%)", fontsize=11)
ax2.set_title("Loss ratio across retention targets", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("Motor renewal book — Q2 2026 rate action", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

**Reading the frontier plot.** The left panel shows the classical trade-off: protecting retention costs profit. The knee marks where the cost accelerates. The right panel confirms that the LR target is met (or approximately met) across the feasible frontier — when retention is forced tighter, the optimiser cannot fully close the LR gap and the LR rises slightly above target.

The question for the pricing committee is not "what retention floor do we use?" It is "we are at the knee at 85%; pushing to 87% costs £X in profit. Is that worth the extra customer protection?" The committee can now answer with numbers, not intuition.
