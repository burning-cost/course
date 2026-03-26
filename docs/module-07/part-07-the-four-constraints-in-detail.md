## Part 7: The four constraints in detail

Now we build the full constraint configuration and optimiser. Spend time understanding what each constraint does before moving on.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig
import numpy as np

# ---- Constraint parameters ----
LR_TARGET       = 0.72    # close the 3pp LR gap from 75% to 72%
RETENTION_FLOOR = 0.85    # accept at most 15% retention loss
MAX_RATE_CHANGE = 0.20    # ±20% year-on-year rate movement cap

config = ConstraintConfig(
    lr_max=LR_TARGET,
    retention_min=RETENTION_FLOOR,
    max_rate_change=MAX_RATE_CHANGE,
    enbp_buffer=0.0,
    technical_floor=True,
)

# ---- Optimiser ----
opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=np.ones(N),
    constraints=config,
    demand_model="log_linear",
)
```

### Constraint 1: lr_max (loss ratio ceiling)

This says: at the new rates, the expected portfolio loss ratio must be at or below 72%.

The expected LR is:

```
E[LR(m)] = sum_i(expected_loss_cost_i * demand_i(m)) / sum_i(technical_price_i * m_i * demand_i(m))
```

The denominator includes `demand_i(m)` — the demand probability at the new multiplier. If you raise rates, some customers will not renew, and their premium drops out of the denominator. This changes the LR in ways that depend on whether lapsing customers are better or worse risks than the average.

This is why you cannot compute the new LR by applying rate changes to a static book — you must model the demand response. The optimiser does this automatically.

### Constraint 2: retention_min (volume floor)

This says: the expected fraction of renewal customers who remain at new rates must be at or above 85%.

```
E[retention] = sum_i(demand_i(m) | renewal_i=True) / n_renewals >= 0.85
```

The 85% floor is a commercial decision: how much volume are you willing to lose in exchange for LR improvement? Setting a tighter floor (e.g., 90%) gives the optimiser less room to take rate. Setting a looser floor (e.g., 80%) allows more aggressive rate action.

The efficient frontier (Part 9) traces the profit-retention trade-off across the full range. You should show the frontier to the pricing committee rather than picking a single threshold number in isolation.

### Constraint 3: ENBP (FCA PS21/11)

This enforces that every renewal customer's new premium does not exceed the equivalent new business price. It is implemented as per-policy multiplier upper bounds:

```
m_i <= enbp_i / technical_price_i    for all renewal policies i
```

By operating in multiplier space, the bound is dimensionless and consistent across policies of different sizes. The constraint is enforced as a scipy `Bounds` object, which SLSQP handles more efficiently than an inequality constraint.

The `enbp_buffer` parameter adds a safety margin. With `enbp_buffer=0.01`, the upper bound becomes `enbp_i * 0.99 / technical_price_i` — a 1% cushion below ENBP for floating-point safety.

A note on PS21/11 in practice: the ENBP calculation requires careful alignment with your actuarial pricing model. Work with your compliance team to confirm which factors are treated as renewal-specific when computing the ENBP. The `enbp` array you pass to the optimiser must be computed correctly upstream — the library enforces the constraint exactly as given.

### Constraint 4: max_rate_change (per-policy rate cap)

This enforces the underwriting committee's approved movement cap. With `max_rate_change=0.20`:

```
(1 - 0.20) * prior_multiplier_i <= m_i <= (1 + 0.20) * prior_multiplier_i
```

This is also implemented as multiplier bounds, not an inequality constraint. The `prior_multiplier` array must be passed to the optimiser — if you omit it, it defaults to 1.0 for all policies (meaning the cap is ±20% from the technical price, not from the prior year's rate).

Setting `prior_multiplier = current_premium / technical_price` gives the cap relative to current market rates, which is the usual convention for year-on-year rate tracking.

**When do the caps cause infeasibility?** If the book needs more rate than the caps allow, the problem is infeasible. The solver will return `result.converged = False`. The remedy is either wider caps (escalate to underwriting director), a looser LR target, or a looser retention floor.

### Why SLSQP for this problem

SLSQP (Sequential Least Squares Programming) is the solver from `scipy.optimize`. It handles nonlinear inequality constraints (the LR and retention constraints are nonlinear because demand enters through the log-linear or logistic function) and box constraints (ENBP and rate change bounds) in a single unified framework.

For N=5,000 policies with analytical gradients, a single solve takes a few seconds. The library provides analytical gradients for all constraints — without them, SLSQP uses finite differences, which is 2*N extra function evaluations per iteration and prohibitively slow for large portfolios.

For larger problems (N=50,000+), the same solver works — analytical gradients keep the per-iteration cost linear in N. The main bottleneck becomes memory: the Jacobian matrix is O(N) dense at each iteration step.
