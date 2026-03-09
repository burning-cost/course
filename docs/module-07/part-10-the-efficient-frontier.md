## Part 10: The efficient frontier

A single solve gives you one point: the factor adjustment vector that achieves 72% LR at minimum dislocation. The efficient frontier gives you the full curve: all achievable (LR, volume) combinations across a range of LR targets.

This is the tool for the pricing committee conversation. Instead of asking "should we take 72% or 71%?", you ask "here is the frontier; which point on it do we want to operate at?"

### Tracing the frontier

```python
from rate_optimiser import EfficientFrontier
import matplotlib.pyplot as plt

frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=25)

# frontier_df is a pandas DataFrame with one row per LR target
print(frontier_df[["lr_target", "expected_lr", "expected_volume",
                    "shadow_lr", "shadow_volume", "feasible"]].to_string(index=False))
```

**What you should see** (abbreviated):

```sql
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume  feasible
      0.68        0.680            0.951     0.2841         0.0000     True
      0.69        0.690            0.958     0.2103         0.0000     True
      0.70        0.700            0.963     0.1512         0.0000     True
      0.71        0.710            0.969     0.1201         0.0000     True
      0.72        0.720            0.973     0.0943         0.0001     True
      0.73        0.730            0.978     0.0712         0.0000     True
      0.74        0.740            0.982     0.0521         0.0000     True
      0.75        0.751            1.000     0.0000         0.0000     True
      0.76        0.751            1.000     0.0000         0.0000     True
      0.77        0.751            1.000     0.0000         0.0000     True
```

The rows at 0.75-0.77 show that once the LR target is loose enough (above the current LR of 0.75), no rate change is needed and the solution is to hold all factors at 1.0. These are informative: they tell you the frontier's right endpoint.

At tight LR targets (0.68, 0.69), the volume retention drops noticeably and the shadow price rises. At 0.68%, volume retention is only 95.1% — below the 97% floor. The feasibility flag says True, but the volume constraint is satisfied because we are tracing the frontier without the volume constraint to show the full unconstrained trade-off. In practice, 0.68% would not be achievable with a 97% volume floor.

### Understanding shadow prices

The `shadow_lr` column is the most important for the pricing committee conversation. It is the Lagrange multiplier on the LR constraint.

The Lagrange multiplier has a precise economic meaning: it is the marginal increase in total dislocation (objective value) per unit relaxation of the LR bound.

In plain English: if the LR target is 0.72 and the shadow price is 0.094, then relaxing the target by 1pp — accepting 73% instead of 72% — would reduce the total dislocation by approximately 0.094 units. Or equivalently, tightening from 72% to 71% would cost approximately 0.094 additional units of dislocation.

This is what you want to show a commercial director. Not "the factor adjustments are 3.7%", but "the cost of improving LR by one more percentage point is X units of customer disruption, and you can see from this table exactly how that cost escalates as you push harder."

At loose LR targets (0.75, 0.76), the shadow price is zero: the constraint is not binding, no rate action is needed, and relaxing it further costs nothing. As the target tightens, the shadow price rises. The rate at which it rises tells you how quickly you are entering diminishing returns.

### Identifying the knee of the frontier

The knee of the frontier is where the shadow price starts rising faster than the LR improvement justifies. We define it as the point where the shadow price first exceeds twice its value at the tightest feasible target in the upper range:

```python
# Filter to feasible rows and those where volume stays above the floor
feasible = frontier_df[
    frontier_df["feasible"] & (frontier_df["expected_volume"] >= 0.97)
].copy().reset_index(drop=True)

# Shadow price at the loosest feasible target (the "cheap" end of rate-taking)
shadow_start = feasible["shadow_lr"].min()

# Knee: first point where shadow price exceeds 2x the starting value
knee_rows = feasible[feasible["shadow_lr"] >= 2 * shadow_start]

if not knee_rows.empty:
    knee_row = knee_rows.iloc[-1]  # tightest target where shadow price is still below 2x
    print(f"Knee of the efficient frontier:")
    print(f"  LR target:    {knee_row['lr_target']:.3f}")
    print(f"  Expected LR:  {knee_row['expected_lr']:.3f}")
    print(f"  Volume:       {knee_row['expected_volume']:.3f}")
    print(f"  Shadow price: {knee_row['shadow_lr']:.4f}")
    print(f"  Shadow price is {knee_row['shadow_lr'] / shadow_start:.1f}x the starting value")
else:
    print("No clear knee found in feasible range — extend the LR range.")
```

### Plotting the frontier for the pricing committee

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: LR vs volume retention
feasible_vol = frontier_df[
    frontier_df["feasible"] & (frontier_df["expected_volume"] >= 0.95)
]
ax1.plot(
    feasible_vol["expected_lr"] * 100,
    feasible_vol["expected_volume"] * 100,
    "o-", color="steelblue", linewidth=2, markersize=5,
)
# Mark the knee
if not knee_rows.empty:
    ax1.scatter(
        [knee_row["expected_lr"] * 100],
        [knee_row["expected_volume"] * 100],
        color="firebrick", s=100, zorder=5, label="Knee",
    )
ax1.axhline(97, linestyle="--", color="grey", alpha=0.5, label="Volume floor (97%)")
ax1.set_xlabel("Expected loss ratio (%)", fontsize=11)
ax1.set_ylabel("Expected volume retention (%)", fontsize=11)
ax1.set_title("Efficient frontier: LR vs volume", fontsize=12)
ax1.invert_xaxis()
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right panel: shadow price vs LR target
ax2.plot(
    feasible_vol["lr_target"] * 100,
    feasible_vol["shadow_lr"],
    "o-", color="darkorange", linewidth=2, markersize=5,
)
if not knee_rows.empty:
    ax2.axhline(
        2 * shadow_start,
        linestyle="--", color="firebrick", alpha=0.6,
        label=f"2x initial shadow price ({2*shadow_start:.4f})",
    )
ax2.set_xlabel("LR target (%)", fontsize=11)
ax2.set_ylabel("Shadow price on LR constraint", fontsize=11)
ax2.set_title("Marginal cost of LR improvement", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("Motor renewal book — Q2 2026 rate action", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

**Reading the frontier plot.** The left panel shows the classical trade-off: improving LR costs volume. The knee (red dot) is the natural stopping point — beyond it, each additional percentage point of LR improvement costs disproportionately more volume. The right panel shows the shadow price rising sharply at tight LR targets: this is the direct quantification of what the left panel shows graphically.

The question for the pricing committee is not "should we take 72% or 71%?" It is "we are currently at the knee at 72%; pushing to 71% costs an additional 0.094 units of dislocation per pp. Is that worth the extra LR headroom?" The committee can now answer with numbers, not intuition.