## Part 9: Solving the problem

Run the solver:

```python
result = opt.optimise()

print(result)
print()
print(f"Converged:          {result.converged}")
print(f"Solver message:     {result.solver_message}")
print(f"Iterations:         {result.n_iter}")
print()
print(f"Expected profit:    £{result.expected_profit:,.0f}")
print(f"Expected GWP:       £{result.expected_gwp:,.0f}")
print(f"Expected LR:        {result.expected_loss_ratio:.4f}  (target: {LR_TARGET})")
print(f"Expected retention: {result.expected_retention:.4f}  (floor: {RETENTION_FLOOR})")
print()
print("Multiplier distribution:")
print(f"  Mean:   {result.multipliers.mean():.4f}")
print(f"  Median: {np.median(result.multipliers):.4f}")
print(f"  10th:   {np.percentile(result.multipliers, 10):.4f}")
print(f"  90th:   {np.percentile(result.multipliers, 90):.4f}")
```

**What you should see:**

```text
OptimisationResult(CONVERGED, N=5000, profit=1,456,123, gwp=7,012,345, lr=0.720)

Converged:          True
Solver message:     Optimization terminated successfully
Iterations:         87

Expected profit:    £1,456,123
Expected GWP:       £7,012,345
Expected LR:        0.7200  (target: 0.72)
Expected retention: 0.8503  (floor: 0.85)

Multiplier distribution:
  Mean:   1.0368
  Median: 1.0342
  10th:   0.9821
  90th:   1.0812
```

Exact values depend on the random seed, but the LR should hit the target (within solver tolerance) and the retention should be at or just above the floor.

### Reading the result

**`result.converged`** must be `True` before you use any other output. If it is `False`, the solver failed and the multipliers are not a valid solution. See below for what to do.

**`result.expected_profit`** is the objective: `sum((price - cost) * demand)` at the optimal multipliers. This is what the solver maximised.

**`result.expected_loss_ratio`** is the expected portfolio LR at the new rates, after accounting for demand-driven lapses. It should be at or just below the LR target — the constraint is binding.

**`result.expected_retention`** is the expected renewal retention rate. A value of 0.8503 means the optimiser expects 14.97% of renewal customers to lapse due to rate-driven increases. The retention constraint is (slightly) binding.

**`result.multipliers`** is the array of optimal price multipliers, shape `(N,)`. The final premium for policy `i` is `result.multipliers[i] * technical_price[i]`.

**`result.new_premiums`** is the array of optimal final premiums, shape `(N,)`. Equivalent to `result.multipliers * technical_price`.

**`result.shadow_prices`** is a dict of Lagrange multipliers on the active constraints. See Part 10 for how to read these.

**`result.summary_df`** is a per-policy Polars DataFrame with columns: `policy_idx`, `multiplier`, `new_premium`, `expected_demand`, `contribution`, `enbp_binding`, `rate_change_pct`.

### Why policies move differently

Unlike a uniform factor-table adjustment, the per-policy solver produces heterogeneous rate changes. The key driver is the per-policy elasticity:

- Low-elasticity (direct, tenured) customers are less likely to lapse when rates go up. The optimiser can push their multipliers higher without violating the retention constraint.
- High-elasticity (PCW, new) customers are very price-sensitive. The optimiser keeps their multipliers closer to 1.0 to avoid triggering lapses that would breach the retention floor.

Within the constraint set, the objective is to maximise profit. High-elasticity customers are worth less in profit terms per unit of rate increase because their demand falls faster. The optimiser naturally differentiates.

### When the solver does not converge

If `result.converged` is `False`, the causes are almost always:

**Infeasibility.** The constraints cannot all be satisfied simultaneously. Use the efficient frontier (Part 10) to find a feasible region. Loosen the retention floor or widen the rate change cap.

**Near-infeasibility.** The problem is technically feasible but SLSQP's iteration limit runs out before convergence. Increase the maximum iterations: `PortfolioOptimiser(..., maxiter=2000)`. Or use `n_restarts=3` to try from multiple starting points.

**Conditioning.** If elasticities are very large in magnitude (e.g., -10.0), small multiplier changes cause large demand swings and the gradient becomes noisy. Check that elasticities are in a plausible range (-0.5 to -3.5 for personal lines).

The correct response to non-convergence is always to investigate the cause before relaxing constraints. Never present results from a non-converged solve.
