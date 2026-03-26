## Part 6: Configuring the constraints and building the optimiser

The `insurance-optimise` library does not wrap data in validator objects — it takes numpy arrays directly. All constraint configuration goes into a single `ConstraintConfig` dataclass, which you then pass to `PortfolioOptimiser`.

### ConstraintConfig

`ConstraintConfig` is a dataclass that declares every active constraint in one place. None of the fields are required — unset fields simply mean unconstrained.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.72,             # aggregate LR must not exceed 72%
    retention_min=0.85,      # expected renewal retention must be at least 85%
    max_rate_change=0.20,    # per-policy rate change capped at ±20%
    enbp_buffer=0.0,         # tight to ENBP (no extra safety margin)
    technical_floor=True,    # prices must be at or above technical price
)

print(config)
```

The ENBP constraint is not a separate object. It is enforced automatically via per-policy multiplier upper bounds when you pass `enbp` and `renewal_flag` arrays to `PortfolioOptimiser`. The `enbp_buffer` parameter adds a small safety margin below the ENBP: setting `enbp_buffer=0.01` means the renewal premium upper bound is `enbp * 0.99` rather than `enbp` exactly.

### Building the optimiser

```python
opt = PortfolioOptimiser(
    technical_price=technical_price,    # shape (N,)
    expected_loss_cost=expected_loss_cost,  # shape (N,)
    p_demand=p_demand,                  # shape (N,), values in (0,1)
    elasticity=elasticity,              # shape (N,), negative
    renewal_flag=renewal_flag,          # shape (N,), boolean
    enbp=enbp,                          # shape (N,), positive
    prior_multiplier=np.ones(N),        # optional: prior year multipliers
    constraints=config,
    demand_model="log_linear",
    solver="slsqp",
    seed=42,
)

print(f"Built: {opt.n} policies, {opt.n_constraints} portfolio constraints")
```

**What you should see:**

```bash
Built: 5,000 policies, 2 portfolio constraints
```

The 2 portfolio constraints are LR and retention. ENBP and rate change bounds appear as box constraints on the multipliers (not counted here) — SLSQP handles them separately as bounds, which is more efficient than inequality constraints.

### Checking baseline metrics

Before solving, call `portfolio_summary()` to confirm the starting point:

```python
baseline = opt.portfolio_summary(m=np.ones(N))
print(f"Baseline LR:        {baseline['loss_ratio']:.4f}  (target: {config.lr_max:.4f})")
print(f"Baseline retention: {baseline['retention']:.4f}  (floor: {config.retention_min:.4f})")
print(f"Baseline profit:    £{baseline['profit']:,.0f}")
print(f"LR gap to close:    {(baseline['loss_ratio'] - config.lr_max)*100:.1f}pp")
```

**What you should see:**

```bash
Baseline LR:        0.7500  (target: 0.7200)
Baseline retention: 0.7612  (floor: 0.8500)
Baseline profit:    £1,234,567
LR gap to close:    3.0pp
```

The LR constraint is violated at current rates — that is expected. The retention constraint is satisfied — at current rates, we have not taken any rate action to trigger lapses. The solver needs to find multipliers that close the 3pp LR gap while keeping retention above 85%.

### Why per-policy multipliers, not factor-table adjustments?

The classic approach to rate optimisation solves for a small number of factor-level adjustments (one multiplier per rating factor). `insurance-optimise` instead solves for a multiplier per policy.

The reasons are:

1. **Richer solutions.** A per-policy solve can differentiate between customers who are at very different points on the demand curve, even within the same factor level. Two 25-year-old drivers in the same NCB and vehicle group may have very different retention probability histories; the per-policy approach can reflect this.

2. **Correct ENBP handling.** The ENBP constraint is inherently per-policy (each renewal has its own ENBP from PS21/11). Encoding it as a per-policy multiplier bound is exact; expressing it as a factor-level bound requires approximation.

3. **Consistent with modern pricing.** In a GBM/neural-network pricing world, the "factor table" abstraction is less useful — risk is continuous, not bucketed. Per-policy optimisation is the natural complement.

The tradeoff: the solution is harder to communicate to the rating engine (N multipliers rather than K factor adjustments). In practice, you post-process the per-policy multipliers into segment-level adjustments for implementation. See Part 11 for how to do this.
