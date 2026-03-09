## Part 9: Solving the problem

Run the solver:

```python
result = opt.solve()

print(f"Converged:         {result.converged}")
print(f"Objective value:   {result.objective_value:.6f}")
print(f"Expected LR:       {result.expected_loss_ratio:.4f}")
print(f"Expected volume:   {result.expected_volume_ratio:.4f}")
print()
print("Factor adjustments:")
print(f"  {'Factor':<25} {'Multiplier':>12} {'Change':>10} {'Direction':>12}")
print(f"  {'-'*61}")
for factor, m in result.factor_adjustments.items():
    direction = "up" if m > 1.0 else ("down" if m < 1.0 else "unchanged")
    print(f"  {factor:<25} {m:>12.4f} {(m-1)*100:>+9.1f}%  {direction:>12}")
```

**What you should see:**

```text
Converged:         True
Objective value:   0.006832
Expected LR:       0.7200
Expected volume:   0.9731

Factor adjustments:
  Factor                      Multiplier     Change    Direction
  -------------------------------------------------------------
  f_age                           1.0368    +3.7%          up
  f_ncb                           1.0361    +3.6%          up
  f_vehicle                       1.0355    +3.6%          up
  f_region                        1.0359    +3.6%          up
  f_tenure_discount               1.0000    +0.0%   unchanged
```

The exact values will vary slightly with different random seeds, but the pattern should be:
- All four shared factors increase by approximately 3.5-4%
- The tenure discount is unchanged (ENBP constraint prevents it from increasing)

### Reading the result

**`result.converged`** must be True before you use any other output. If it is False, the solver failed and the factor adjustments are not a valid solution. See Part 10 for what to do.

**`result.objective_value`** is the total dislocation: the sum of squared deviations from 1.0. For five factors each at approximately 1.037, this is roughly 5 x (0.037)^2 = 0.0068. Lower is better — it means the rate action is smaller.

**`result.expected_loss_ratio`** is the LR the optimiser expects at the new rates, after accounting for demand-driven lapses. It should be at or very close to the LR target. If it is materially above the target, the LR constraint was binding and the optimiser could not quite reach it — which should not happen if the problem was feasible.

**`result.expected_volume_ratio`** is the expected retention ratio. A value of 0.973 means the optimiser expects 2.7% volume loss from rate-driven lapses. Since the floor is 97%, the volume constraint is satisfied (barely — the constraint is nearly binding).

**`result.factor_adjustments`** is the dictionary of m\_k values. This is the deliverable: the rate action to present to the pricing committee and implement in the rating engine.

### Why all factors move by similar amounts

The minimum-dislocation objective penalises large deviations equally across all factors. With no other asymmetry in the problem (same bounds on all factors, no preference for one factor over another), the optimiser spreads the rate increase evenly. A 3.7% increase on five factors gives the same total premium change as a 18.5% increase on one factor, but the dislocation is 5 x (0.037)^2 = 0.0068 vs (0.185)^2 = 0.034 — five times larger. The solver strongly prefers the spread.

The tenure discount stays at 1.0 because the ENBP constraint prevents it from moving above 1.0. This means the rate increase is shared entirely across the four shared factors. If ENBP were not a constraint (which it is, but hypothetically), the optimiser might have spread some increase to the tenure discount as well, and the increase on the other four factors would be slightly smaller.

### When the solver does not converge

If `result.converged` is False, the causes are almost always one of three things:

**Infeasibility.** The constraints cannot all be satisfied simultaneously. Check the feasibility report. If the volume floor is 97% and the LR target requires more rate than the demand model allows without breaching the volume floor, the problem is infeasible. Loosen the volume floor or widen the factor caps.

**Near-infeasibility.** The problem is technically feasible but SLSQP's iteration limit runs out before convergence. Increase the maximum iterations: `opt.solve(max_iter=2000)`. Or relax the tightest constraint slightly, solve, then gradually re-tighten.

**Demand model conditioning.** If the price coefficient in the logistic model is very large in magnitude (say, -10.0), small changes in the rate vector cause large changes in renewal probability, and the gradient used by SLSQP becomes noisy. Check your demand model parameters are in a plausible range before solving.

The correct response to non-convergence is always to investigate the cause before relaxing constraints. Never present results from a non-converged solve. The output is not a valid solution.