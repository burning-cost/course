## Part 8: Checking the baseline before solving

Before running the solver, always verify the starting point. Call `portfolio_summary()` to confirm the current metrics and understand how far the book is from each constraint:

```python
baseline = opt.portfolio_summary(m=np.ones(N))

print("Baseline metrics (at current rates, m=1.0 for all policies):")
print(f"  LR:        {baseline['loss_ratio']:.4f}  (target: {LR_TARGET:.4f})")
print(f"  Retention: {baseline['retention']:.4f}  (floor: {RETENTION_FLOOR:.4f})")
print(f"  Profit:    £{baseline['profit']:,.0f}")
print(f"  GWP:       £{baseline['gwp']:,.0f}")
print()
print(f"LR gap to close:         {(baseline['loss_ratio'] - LR_TARGET)*100:.1f}pp")
print(f"Retention headroom:      {(baseline['retention'] - RETENTION_FLOOR)*100:.1f}pp")
```

**What you should see:**

```text
Baseline metrics (at current rates, m=1.0 for all policies):
  LR:        0.7500  (target: 0.7200)
  Retention: 0.7612  (floor: 0.8500)
  Profit:    £1,234,567
  GWP:       £6,789,012

LR gap to close:         3.0pp
Retention headroom:      -8.9pp
```

The LR constraint is violated at current rates — expected, it is why you are running a rate action. The retention is below the floor too — this is also expected at m=1.0 because the baseline demand (`p_demand`) represents current renewal rates, which are below 85% for some policies. The solver's job is to find multipliers that satisfy both constraints simultaneously.

**Note on the retention baseline.** The retention floor applies at the *optimised* rates. The baseline retention at m=1.0 can be below the floor — it just means the current portfolio has some retention issues that are not caused by the rate action. The constraint says: after the rate action, retention must be at least 85%. Not: the current retention must be at least 85%.

### Understanding feasibility

There is no separate `feasibility_report()` method. Instead, the solver tells you whether a feasible solution exists: if `result.converged = True`, the constraints are simultaneously satisfiable and the solver found the optimum. If `result.converged = False`, either:

1. The problem is genuinely infeasible (no multiplier vector satisfies all constraints)
2. The solver ran out of iterations before convergence (increase `maxiter`)
3. The problem is nearly infeasible (tighten constraints slightly to make it well-conditioned)

The most common cause of infeasibility is the LR and retention constraints being simultaneously too tight. The efficient frontier (Part 9) reveals whether a feasible region exists at all.

**When do you have a feasibility problem?** If you need 5pp of LR improvement and the rate change cap only allows +10% multipliers, but your demand model predicts that a +10% rate causes a 20% volume loss (pushing below the retention floor), the problem is infeasible. The three levers are:

1. **Widen the rate change cap**: change `max_rate_change` from 0.10 to 0.20 — gives more rate-taking capacity per policy
2. **Loosen the retention floor**: change `retention_min` from 0.85 to 0.80 — accepts more lapse in exchange for LR
3. **Accept a less ambitious LR target**: present the frontier to the pricing committee and let them choose a feasible point

Never relax the ENBP constraint. It is a regulatory requirement.
