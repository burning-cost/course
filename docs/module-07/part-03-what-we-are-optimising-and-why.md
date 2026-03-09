## Part 3: What we are optimising and why

Before writing any optimisation code, you need to understand the three elements of any constrained optimisation problem: the decision variables, the objective function, and the constraints. We explain each in plain English before introducing the maths.

### The decision variables: factor multipliers

Your rating system produces a premium by multiplying a base rate by a series of factors:

```
premium = base_rate x age_factor x ncb_factor x vehicle_factor x region_factor x tenure_discount
```

Each factor is a table. The age factor, for example, looks like:

| Age band | Relativity |
|----------|-----------|
| 17-21    | 2.00      |
| 22-24    | 1.50      |
| 25-29    | 1.20      |
| 30-39    | 1.00      |
| 40-54    | 0.92      |
| 55-69    | 0.95      |
| 70+      | 1.10      |

A 19-year-old gets a factor of 2.00. A 45-year-old gets 0.92. These relativities capture the shape of the risk — how much more expensive young drivers are relative to middle-aged drivers.

In a **uniform rate action**, the shape of the table does not change. Instead, you scale the entire table by a single multiplier. If the age factor adjustment is 1.038, then every level of the age table increases by 3.8%:

| Age band | Old relativity | New relativity | Change |
|----------|---------------|----------------|--------|
| 17-21    | 2.00          | 2.076          | +3.8%  |
| 22-24    | 1.50          | 1.557          | +3.8%  |
| 25-29    | 1.20          | 1.246          | +3.8%  |

The decision variables in the optimisation are these multipliers: one per factor. We write them as a vector **m** = (m\_age, m\_ncb, m\_vehicle, m\_region, m\_tenure).

For five factors, we are solving for five numbers. That is a small optimisation problem — one reason it runs in under a second on a laptop.

### The objective function: minimum dislocation

Given that there are many vectors **m** that could achieve the LR target, which one should you pick?

The principle is minimum dislocation: choose the rate action that achieves the target while changing premiums as little as possible. This is both a commercial principle (unhappy customers lapse) and a Consumer Duty principle (disproportionate increases on specific segments need justification).

Mathematically, the objective function is the sum of squared deviations of the multipliers from 1.0:

```
minimise:  sum_k (m_k - 1)^2
```

where k runs over the five factors. This is called the minimum-dislocation objective.

Why squared? Two reasons.

First, squaring makes large deviations much more costly than small ones. A 10% increase on one factor contributes (0.10)^2 = 0.01 to the objective. Two 5% increases on two factors contribute 2 x (0.05)^2 = 0.005. The solver will prefer spreading the rate increase across factors rather than concentrating it on one, because concentrating produces a much larger objective value.

Second, squaring makes the objective function convex. This is a mathematical property that guarantees there is exactly one minimum within the feasible set — there is no risk of the solver finding a local minimum that is not the global minimum. For pricing, this is important: the solution is unique and reproducible.

If you used absolute deviations instead of squared deviations, the objective would be convex but not strictly convex, and there could be multiple solutions with the same objective value. The solver might return different solutions on different runs. With squared deviations, the solution is always unique.

### The constraints: what must be satisfied

The constraints are the conditions the factor vector **m** must satisfy. There are four.

**Constraint 1: Loss ratio target.** The expected portfolio loss ratio at the new rates must be at or below the target:

```
E[LR(m)] <= LR_target
```

The expected LR is not simply the current LR divided by the average rate change, because some customers will lapse when you raise rates. Lapsed customers do not contribute expected losses or expected premium to the renewed book. The LR calculation must account for this through the demand model.

**Constraint 2: Volume floor.** The expected volume retained at the new rates must be at or above the floor:

```
E[volume(m)] >= volume_floor
```

Volume is measured as expected retained premium at new rates divided by expected retained premium at current rates. A 97% floor means you are willing to accept at most 3% volume loss from rate-driven lapses.

**Constraint 3: ENBP (PS 21/5).** For every renewal policy on every relevant channel, the adjusted renewal premium must not exceed the new business equivalent premium. The FCA's PS 21/5, effective January 2022, requires this at the individual policy level — not just on average.

**Constraint 4: Factor movement caps.** Each adjustment m\_k must lie within the range approved by the underwriting committee. If the caps are 90% to 115%, then:

```
0.90 <= m_k <= 1.15  for all k
```

These four constraints define the feasible set: the region of the (m\_1, m\_2, ..., m\_F) space where all constraints are simultaneously satisfied. The solver finds the point in this region with the smallest objective value.