## Part 7: The four constraints in detail

Now we build the optimiser and add the four constraints. Each constraint is an object you add to the optimiser. Spend time understanding what each one does before moving on.

```python
from rate_optimiser import (
    RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# ---- Parameters ----
LR_TARGET    = 0.72    # close the 3.2pp gap from 75.2% to 72.0%
VOLUME_FLOOR = 0.97    # accept at most 3% volume loss from rate-driven lapses
FACTOR_LOWER = 0.90    # no factor can decrease by more than 10%
FACTOR_UPPER = 1.15    # no factor can increase by more than 15%

# ---- Demand model ----
params = LogisticDemandParams(
    intercept=1.2,
    price_coef=-2.0,   # log-price semi-elasticity for this book
    tenure_coef=0.05,  # stickiness per year of tenure
)
demand = make_logistic_demand(params)

# ---- Optimiser ----
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(
    lower=FACTOR_LOWER,
    upper=FACTOR_UPPER,
    n_factors=fs.n_factors,
))
```

### Constraint 1: LossRatioConstraint

This says: at the new rates, the expected portfolio loss ratio must be at or below 72%.

The calculation is more subtle than it first appears. The expected LR is:

```python
E[LR(m)] = sum_i(expected_claims_i) / sum_i(expected_premium_i x renewal_prob_i(m))
```

The denominator includes `renewal_prob_i(m)` — the probability of policy i renewing at the new rates given the factor adjustment m. If you raise rates substantially, some policies will not renew, and their premium drops out of the denominator. This can move the LR up or down depending on the risk profile of the lapsing policies.

If the customers most likely to lapse are the better risks (lower technical premium relative to current premium), their departure improves the LR. If the customers most likely to lapse are the worse risks (higher technical premium relative to current premium), their departure worsens the LR. The demand model determines which type is more sensitive to price.

This is why you cannot ignore the demand model when projecting LR at new rates.

### Constraint 2: VolumeConstraint

This says: the expected volume at new rates, measured as retained premium, must be at or above 97% of current volume.

```sql
E[sum_i(premium_i(m) x renewal_prob_i(m))] >= 0.97 x E[sum_i(premium_i(current) x renewal_prob_i(current))]
```

The 97% floor is a commercial decision: how much volume are you willing to lose in exchange for LR improvement? Setting a tighter floor (e.g., 99%) gives the optimiser less room to take rate. Setting a looser floor (e.g., 95%) allows more aggressive rate action.

The exercises explore the trade-off. A key insight from the efficient frontier (Part 9) is that the relationship between the volume floor and achievable LR is nonlinear: relaxing the floor from 97% to 96% may give you much more LR headroom than relaxing from 96% to 95%.

### Constraint 3: ENBPConstraint

This enforces PS 21/5 compliance at the individual policy level. The library evaluates:

```sql
for every renewal policy i in channels ["PCW", "direct"]:
    adjusted_renewal_i <= NB_equivalent_i
```

The constraint is satisfied if and only if this holds for every renewal policy. The library implements this as a maximum-excess constraint: `max_i(adjusted_renewal_i - NB_equivalent_i) <= 0`. This is mathematically equivalent to a per-policy constraint but computationally tractable.

A note on PS 21/5 in practice: the ENBP calculation requires careful alignment with your actuarial pricing model. Work with your compliance team to confirm which factors are treated as renewal-specific when computing the NB equivalent. The computation must account for introductory discounts that new business customers receive, channel-specific underwriting appetite (some insurers offer different terms to NB on PCW versus direct), and any loyalty adjustments applied to renewals. The `renewal_factor_names` parameter handles the structural part of this, but the commercial layer requires human review.

### Constraint 4: FactorBoundsConstraint

This enforces the underwriting committee's approved movement caps. Setting them to [0.90, 1.15] means:

- No factor can decrease by more than 10%
- No factor can increase by more than 15%

These caps serve two purposes. First, they reflect underwriting risk management: an age factor that jumps by 30% in a single cycle creates adverse selection risk and disrupts the book in ways that are difficult to reverse. Second, they create a principled stopping point: if the problem cannot be solved within the caps, you need to either relax a constraint or escalate to the underwriting director for a wider mandate.

**When do the caps cause infeasibility?** If you are 5pp above LR target and the factor caps only allow 5% increases, but your demand model predicts that a 5% increase causes a 4% volume loss (pushing you below the volume floor), the problem is infeasible within the caps. The feasibility check in Part 8 reveals this before you waste time on a failed solve.

### Why SLSQP for this problem

SLSQP (Sequential Least Squares Programming) is the solver from `scipy.optimize`. It handles nonlinear inequality constraints (the LR and volume constraints are nonlinear because renewal probability enters through the logistic function) and box constraints (the factor bounds) correctly.

For this problem size — 5 to 20 factors, 5,000 to 200,000 policies — SLSQP is the right choice. It is efficient for smooth nonlinear problems with a moderate number of variables and constraints. For larger problems with more factors (say, 50+ factor tables), consider `trust-constr` from the same `scipy.optimize` module: it is more robust on problems where the constraint Jacobian is ill-conditioned, at the cost of being slower per iteration.

For problems with hundreds of decision variables (e.g., per-level optimisation of every cell in every factor table), SLSQP and trust-constr both become slow and you would need a different approach — quadratic programming with a KKT-based solver, or a stochastic gradient method. That is a significantly more complex problem; the uniform-factor optimisation in this module is the appropriate starting point for most pricing reviews.