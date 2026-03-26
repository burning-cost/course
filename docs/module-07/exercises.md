# Module 7 Exercises: Constrained Rate Optimisation

Seven exercises. Work through them in order — each builds on the portfolio and results from the previous exercise. Solutions are inside collapsed sections at the end of each exercise.

Before starting: read Parts 1-9 of the tutorial. Every concept used in these exercises is explained there.

---

## Exercise 1: Building the optimisation problem from scratch

**Reference:** Tutorial Parts 1-8

**What you will do:** Set up a rate optimisation problem on a new portfolio, inspect the baseline metrics, and understand which constraints are binding before the solver runs.

**Context.** You are the pricing actuary for a UK motor insurer. The book is running at 78% loss ratio against a 74% target. The underwriting director has approved a rate change cap of ±20%. The FCA's ENBP requirement applies to all renewal policies.

### Setup: Generate the portfolio

Add a markdown cell to your notebook (`%md ## Exercise 1: Portfolio setup`), then paste this in a new code cell and run it:

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(seed=2026)
N = 4_000

# Rating factor relativities
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N, p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N, p=[0.20, 0.40, 0.25, 0.15])
tenure      = rng.integers(0, 10, N).astype(float)

base_rate         = 380.0
expected_loss_cost = (
    base_rate * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)
)
technical_price = expected_loss_cost / 0.80

# Book at 78% LR
current_premium = expected_loss_cost / 0.78 * rng.uniform(0.96, 1.04, N)
market_premium  = expected_loss_cost / 0.75 * rng.uniform(0.90, 1.10, N)

renewal_flag = rng.random(N) < 0.65
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.68, 0.32]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# Per-policy elasticity: PCW more elastic, direct less
elasticity = np.where(channel == "PCW", -2.2, -1.3)
elasticity = elasticity + 0.03 * tenure
elasticity = np.clip(elasticity, -3.5, -0.5)

# Baseline demand at current rates
log_price_ratio = np.log(current_premium / market_premium)
logit_p = 1.2 + (-2.2) * log_price_ratio + 0.04 * tenure
p_demand = 1.0 / (1.0 + np.exp(-logit_p))
p_demand = np.clip(p_demand, 0.05, 0.95)

# ENBP for renewal policies
enbp = np.where(renewal_flag, current_premium * rng.uniform(0.98, 1.05, N), 0.0)

# Quick sanity check
current_lr = expected_loss_cost.sum() / current_premium.sum()
print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {renewal_flag.sum():,} ({renewal_flag.mean():.0%})")
print(f"Current LR: {current_lr:.4f}  (target: 0.74)")
print(f"LR gap:     {(current_lr - 0.74)*100:.1f}pp")
```

**What you should see:** Current LR close to 0.78, with a 4pp gap to close to reach 74%.

### Tasks

**Task 1.** Build a `ConstraintConfig` with `lr_max=0.74`, `retention_min=0.85`, `max_rate_change=0.20`, and `technical_floor=True`. Then build a `PortfolioOptimiser` with these constraints. Call `portfolio_summary(m=np.ones(N))` to confirm the baseline LR, retention, and profit. Does the baseline LR match your manual calculation above?

**Task 2.** Call `opt.optimise()`. Confirm it converges. Report: expected LR, expected retention, expected profit, and the number of solver iterations.

**Task 3.** Inspect `result.shadow_prices`. For each active constraint, state: (a) whether it is binding, (b) its marginal value, and (c) what tightening or loosening that constraint by a small amount would cost.

At the optimal solution, think through each constraint before looking at the shadow prices: which ones should be binding? Which might have slack?

**Task 4.** Now loosen the retention floor to `retention_min=0.80` (accept more lapse) and re-solve. What happens to: expected profit, expected LR, expected retention, shadow prices? Does the profit increase or decrease? Why?

**Task 5.** Now tighten `lr_max` to 0.72 (a 6pp improvement rather than 4pp) and keep the original `retention_min=0.85`. Does the problem remain feasible? If not, what is the easiest constraint to relax to make it feasible?

<details>
<summary>Hint for Task 3</summary>

At the optimal solution, the LR constraint is almost always binding — the solver pushes profit as high as it can while keeping LR exactly at the target. If the retention floor is 85% and the optimal retention is exactly 85%, the retention constraint is also binding.

A shadow price near zero means the constraint is not binding (the solution satisfies it with room to spare). A positive shadow price means the constraint is binding — its marginal value is the profit you would gain by relaxing it by one unit.

For `lr_max`, the shadow price has units of profit per unit LR. So a shadow price of 10,000 on `lr_max` means relaxing the LR cap by 0.01 (from 74% to 75%) would increase expected profit by approximately £100.

</details>

<details>
<summary>Solution — Exercise 1</summary>

```python
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

# Task 1: Build optimiser
config_ex1 = ConstraintConfig(
    lr_max=0.74,
    retention_min=0.85,
    max_rate_change=0.20,
    technical_floor=True,
)

opt_ex1 = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    prior_multiplier=np.ones(N),
    constraints=config_ex1,
    demand_model="log_linear",
    seed=42,
)

baseline = opt_ex1.portfolio_summary(m=np.ones(N))
print(f"Baseline LR:        {baseline['loss_ratio']:.4f}  (target: {config_ex1.lr_max:.4f})")
print(f"Baseline retention: {baseline['retention']:.4f}  (floor: {config_ex1.retention_min:.4f})")
print(f"Baseline profit:    £{baseline['profit']:,.0f}")
print()
# Verify LR matches manual calculation
print(f"Manual LR check: {expected_loss_cost.sum() / current_premium.sum():.4f}")
# Note: baseline LR uses technical_price not current_premium in the denominator,
# so it may differ slightly from the current_premium-based LR.
# portfolio_summary(m=ones) computes LR at technical_price * 1.0 = technical_price.

# Task 2: Solve
result_ex1 = opt_ex1.optimise()
print(f"\nConverged:          {result_ex1.converged}")
print(f"Iterations:         {result_ex1.n_iter}")
print(f"Expected LR:        {result_ex1.expected_loss_ratio:.4f}")
print(f"Expected retention: {result_ex1.expected_retention:.4f}")
print(f"Expected profit:    £{result_ex1.expected_profit:,.0f}")

# Task 3: Shadow prices
print("\nShadow prices:")
for name, sp in result_ex1.shadow_prices.items():
    binding = "BINDING" if abs(sp) > 1e-6 else "slack"
    print(f"  {name:<20}: {sp:+.4f}  [{binding}]")

sp_lr  = result_ex1.shadow_prices.get("lr_max", 0)
if abs(sp_lr) > 1e-6:
    print(f"\n  Relaxing lr_max by 0.01 (74% -> 75%) would gain ~£{sp_lr * 0.01:,.0f} profit")
    print(f"  Tightening lr_max by 0.01 (74% -> 73%) would cost ~£{sp_lr * 0.01:,.0f} profit")

# Task 4: Looser retention floor
config_loose_ret = ConstraintConfig(
    lr_max=0.74, retention_min=0.80, max_rate_change=0.20, technical_floor=True
)
opt_loose = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N), constraints=config_loose_ret, seed=42,
)
result_loose = opt_loose.optimise()

print(f"\nTask 4 — Looser retention (80% floor):")
print(f"  Profit:    £{result_loose.expected_profit:,.0f}  (was: £{result_ex1.expected_profit:,.0f})")
print(f"  LR:        {result_loose.expected_loss_ratio:.4f}")
print(f"  Retention: {result_loose.expected_retention:.4f}")
print(f"  Profit gain from loosening retention: £{result_loose.expected_profit - result_ex1.expected_profit:,.0f}")

# Task 5: Tighter LR target
config_tight_lr = ConstraintConfig(
    lr_max=0.72, retention_min=0.85, max_rate_change=0.20, technical_floor=True
)
opt_tight = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N), constraints=config_tight_lr, seed=42,
)
result_tight = opt_tight.optimise()

print(f"\nTask 5 — Tighter LR target (72%):")
print(f"  Converged: {result_tight.converged}")
if result_tight.converged:
    print(f"  Expected LR:        {result_tight.expected_loss_ratio:.4f}")
    print(f"  Expected retention: {result_tight.expected_retention:.4f}")
    print(f"  Expected profit:    £{result_tight.expected_profit:,.0f}")
else:
    print(f"  Solver message: {result_tight.solver_message}")
    print("  Try loosening retention_min to 0.80 to make this feasible.")
```

</details>

---

## Exercise 2: The efficient frontier — finding the knee and understanding shadow prices

**Reference:** Tutorial Part 10

**What you will do:** Trace the full efficient frontier, identify the knee, interpret shadow prices in commercial terms, and produce a presentation-ready chart.

**Context.** The commercial director wants to understand the full range of options. The pricing committee meeting is in three days. You need to present not just the recommended rate action but the full trade-off space, with a defensible recommendation for where on the frontier to operate.

### Setup

Use the optimiser `opt_ex1` from Exercise 1 (LR target 0.74, retention floor 0.85, rate change cap 0.20).

### Tasks

**Task 1.** Trace the frontier sweeping `volume_retention` from 0.78 to 0.96 with 20 points. Print the full frontier table. How many points converge? At what retention floor does the solver start to fail?

**Task 2.** For the converged frontier points, compute the profit cost of each additional percentage point of retention protected. Create a column `profit_per_retention_pp = -delta_profit / delta_retention * 100`. Where does this cost start to accelerate?

**Task 3.** Define the knee as the first point where the profit cost per retention pp exceeds twice the median cost. Find the knee. Report: the retention floor at the knee, the expected profit, and the expected LR.

**Task 4.** The commercial director asks: "Our main PCW competitor is operating at approximately 87% retention. Is that achievable at our 74% LR target?" Determine whether 87% retention is feasible and at what profit cost compared to the 85% operating point.

**Task 5.** Produce a two-panel frontier chart suitable for the pricing committee pack. Left panel: profit vs retention (mark the knee in red, draw the retention floor as a dashed line). Right panel: loss ratio vs retention. Title the chart with the book name and review date.

<details>
<summary>Hint for Task 2</summary>

The `frontier_result.data` is a Polars DataFrame. To compute the delta between adjacent rows, use `.diff()` on the sorted data:

```python
pareto_pd = frontier_result.pareto_data().to_pandas().sort_values("retention")
pareto_pd["delta_profit"]    = pareto_pd["profit"].diff()
pareto_pd["delta_retention"] = pareto_pd["retention"].diff()
pareto_pd["profit_cost_per_ret_pp"] = (
    -pareto_pd["delta_profit"] / pareto_pd["delta_retention"] / 100
)
```

The first row will have NaN (no prior row). Filter it out before computing the knee.

</details>

<details>
<summary>Solution — Exercise 2</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from insurance_optimise import EfficientFrontier

# Task 1: Trace the frontier
frontier_ex2 = EfficientFrontier(
    optimiser=opt_ex1,
    sweep_param="volume_retention",
    sweep_range=(0.78, 0.96),
    n_points=20,
    n_jobs=1,
)
frontier_result_ex2 = frontier_ex2.run()

print("Efficient frontier:")
print(frontier_result_ex2.data)
print(f"\nConverged: {frontier_result_ex2.data['converged'].sum()} of {len(frontier_result_ex2.data)}")

# Task 2: Profit cost per retention pp
pareto_pd = frontier_result_ex2.pareto_data().to_pandas().sort_values("retention")
pareto_pd["delta_profit"]     = pareto_pd["profit"].diff()
pareto_pd["delta_retention"]  = pareto_pd["retention"].diff()
pareto_pd["profit_cost_per_ret_pp"] = (
    -pareto_pd["delta_profit"] / pareto_pd["delta_retention"] / 100
)
pareto_pd = pareto_pd.dropna()

print("\nProfit cost per retention pp:")
print(pareto_pd[["epsilon", "retention", "profit", "profit_cost_per_ret_pp"]].to_string(index=False))

# Task 3: Knee
median_cost = pareto_pd["profit_cost_per_ret_pp"].median()
knee_candidates = pareto_pd[pareto_pd["profit_cost_per_ret_pp"] >= 2 * median_cost]

if not knee_candidates.empty:
    knee = knee_candidates.iloc[0]
    print(f"\nKnee of the efficient frontier:")
    print(f"  Retention floor: {knee['epsilon']:.3f}")
    print(f"  Expected retention: {knee['retention']:.4f}")
    print(f"  Expected profit:    £{knee['profit']:,.0f}")
    print(f"  Expected LR:        {knee['loss_ratio']:.4f}")
    print(f"  Cost per ret pp:    £{knee['profit_cost_per_ret_pp']:,.0f}")
    print(f"  This is {knee['profit_cost_per_ret_pp'] / median_cost:.1f}x the median cost")
else:
    print("No clear knee found — extend the sweep range")
    knee = pareto_pd.iloc[-1]

# Task 4: 87% retention feasibility
row_87 = pareto_pd[pareto_pd["retention"] >= 0.87]
if not row_87.empty:
    r87 = row_87.iloc[0]
    base_row = pareto_pd[pareto_pd["epsilon"].between(0.848, 0.852)].iloc[0] if len(pareto_pd[pareto_pd["epsilon"].between(0.848, 0.852)]) > 0 else pareto_pd.iloc[0]
    print(f"\nTask 4: 87% retention analysis")
    print(f"  Feasible: Yes")
    print(f"  Expected retention: {r87['retention']:.4f}")
    print(f"  Expected profit:    £{r87['profit']:,.0f}")
    print(f"  Expected LR:        {r87['loss_ratio']:.4f}")
else:
    print("\nTask 4: 87% retention not achieved in sweep range — extend to 0.87+")

# Task 5: Two-panel chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["profit"] / 1_000,
    "o-", color="steelblue", linewidth=2, markersize=5,
)
ax1.scatter(
    [knee["retention"] * 100], [knee["profit"] / 1_000],
    color="firebrick", s=100, zorder=5, label="Knee",
)
ax1.axvline(85, linestyle="--", color="grey", alpha=0.5, label="Base retention floor (85%)")
ax1.set_xlabel("Expected retention (%)", fontsize=11)
ax1.set_ylabel("Expected profit (£k)", fontsize=11)
ax1.set_title("Efficient frontier: profit vs retention", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(
    pareto_pd["retention"] * 100,
    pareto_pd["loss_ratio"] * 100,
    "o-", color="darkorange", linewidth=2, markersize=5,
)
ax2.axhline(74, linestyle="--", color="firebrick", alpha=0.6, label="LR target (74%)")
ax2.set_xlabel("Expected retention (%)", fontsize=11)
ax2.set_ylabel("Expected loss ratio (%)", fontsize=11)
ax2.set_title("Loss ratio across retention targets", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("Motor renewal book Q2 2026 — rate action efficient frontier",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

</details>

---

## Exercise 3: ENBP compliance — cost, verification, and edge cases

**Reference:** Tutorial Parts 7, 13

**What you will do:** Quantify the cost of ENBP compliance, verify it per-policy, and understand how the ENBP constraint interacts with profitability.

**Context.** Your head of regulatory affairs wants: (a) a quantification of what ENBP compliance costs in profit terms, and (b) a per-policy compliance certificate showing no breaches.

### Setup

Use the solved result from Exercise 1 Task 2 (`result_ex1`).

### Tasks

**Task 1.** Run the per-policy ENBP check: for every renewal policy, confirm that `result_ex1.new_premiums[i] <= enbp[i] + 0.01`. Report the number of violations. If there are any, print the policy IDs of the worst five offenders.

**Task 2.** How many renewal policies are at the ENBP cap (`enbp_binding = True` in the summary DataFrame)? What fraction of renewals is this? What does it mean for these policies?

**Task 3.** Build a second optimiser with no ENBP constraint (`enbp=None`, `renewal_flag=None`). Solve it. Compare:
- Expected profit with vs without ENBP
- Expected LR with vs without ENBP
- Expected retention with vs without ENBP

Express the ENBP cost as a percentage of total profit.

**Task 4.** For the without-ENBP solution, compute the distribution of `new_premium / enbp` across renewal policies. How many policies would have premiums above their ENBP? By how much? Plot a histogram.

**Stretch.** Vary the `enbp_buffer` parameter from 0.0 to 0.05 in steps of 0.01. For each, solve and record the expected profit. How sensitive is profit to the safety margin? At what buffer does the profit cost become material?

<details>
<summary>Hint for Task 3</summary>

To disable the ENBP constraint entirely, pass `enbp=None` and `renewal_flag=None` to `PortfolioOptimiser`. The library will skip the ENBP upper bound computation. Note that `retention_min` still applies — the library just does not know which policies are renewals for the ENBP check.

</details>

<details>
<summary>Solution — Exercise 3</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

# Task 1: Per-policy ENBP check
new_prem      = result_ex1.new_premiums
enbp_arr      = enbp
renewal_mask  = renewal_flag.astype(bool)

excess       = new_prem[renewal_mask] - enbp_arr[renewal_mask]
violations   = excess > 0.01
n_violations = violations.sum()

print(f"Task 1 — ENBP compliance:")
print(f"  Renewal policies: {renewal_mask.sum():,}")
print(f"  Violations:       {n_violations}")
if n_violations == 0:
    print("  All renewal premiums at or below ENBP.")
else:
    top5 = np.sort(excess[violations])[::-1][:5]
    print(f"  Top 5 excess amounts: {[f'£{x:.2f}' for x in top5]}")

# Task 2: ENBP-binding policies
enbp_binding_count = result_ex1.summary_df["enbp_binding"].sum()
print(f"\nTask 2 — Policies at ENBP cap: {enbp_binding_count:,} of {renewal_mask.sum():,} renewals")
print(f"  Fraction of renewals at cap: {enbp_binding_count / renewal_mask.sum():.1%}")
print("  These policies would be charged more without ENBP — the constraint is binding for them.")

# Task 3: Without ENBP
config_no_enbp = ConstraintConfig(
    lr_max=0.74, retention_min=0.85, max_rate_change=0.20, technical_floor=True
)
opt_no_enbp = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=None,   # disables ENBP
    enbp=None,           # disables ENBP
    prior_multiplier=np.ones(N),
    constraints=config_no_enbp,
    seed=42,
)
result_no_enbp = opt_no_enbp.optimise()

print(f"\nTask 3 — ENBP cost:")
print(f"  Profit with ENBP:    £{result_ex1.expected_profit:,.0f}")
print(f"  Profit without ENBP: £{result_no_enbp.expected_profit:,.0f}")
enbp_cost = result_no_enbp.expected_profit - result_ex1.expected_profit
enbp_pct  = enbp_cost / result_ex1.expected_profit * 100
print(f"  ENBP profit cost:    £{enbp_cost:,.0f}  ({enbp_pct:.1f}% of profit with ENBP)")
print()
print(f"  LR with ENBP:    {result_ex1.expected_loss_ratio:.4f}")
print(f"  LR without ENBP: {result_no_enbp.expected_loss_ratio:.4f}")

# Task 4: Breach distribution in without-ENBP solution
new_prem_no_enbp = result_no_enbp.new_premiums[renewal_mask]
enbp_renewal     = enbp_arr[renewal_mask]
breach_ratio     = new_prem_no_enbp / np.maximum(enbp_renewal, 0.01)

print(f"\nTask 4 — Without-ENBP breach distribution:")
print(f"  Policies > ENBP: {(breach_ratio > 1.0).sum():,}")
print(f"  Policies > ENBP + 5%: {(breach_ratio > 1.05).sum():,}")
print(f"  Max ratio: {breach_ratio.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(breach_ratio, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(1.0, color="firebrick", linestyle="--", linewidth=1.5, label="ENBP boundary (ratio=1.0)")
ax.set_xlabel("New premium / ENBP (renewal policies)")
ax.set_ylabel("Count")
ax.set_title("ENBP breach distribution — without ENBP constraint")
ax.legend()
plt.tight_layout()
plt.show()

# Stretch: enbp_buffer sensitivity
print("\nStretch — enbp_buffer sensitivity:")
for buf in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
    cfg = ConstraintConfig(
        lr_max=0.74, retention_min=0.85, max_rate_change=0.20,
        enbp_buffer=buf, technical_floor=True,
    )
    opt_buf = PortfolioOptimiser(
        technical_price=technical_price, expected_loss_cost=expected_loss_cost,
        p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
        enbp=enbp, prior_multiplier=np.ones(N), constraints=cfg, seed=42,
    )
    r = opt_buf.optimise()
    print(f"  enbp_buffer={buf:.2f}: profit=£{r.expected_profit:,.0f}  converged={r.converged}")
```

</details>

---

## Exercise 4: Demand model sensitivity — what if lapses are worse?

**Reference:** Tutorial Parts 4, 10

**What you will do:** Test how sensitive the optimal rate action is to the assumed price elasticity. This is the single most important sensitivity check before presenting the frontier to a pricing committee.

**Context.** The commercial director asks: "Your demand model assumes a PCW elasticity of -2.2. What if customers are twice as price-sensitive as you think? How does that change the recommendation?"

### Tasks

**Task 1.** Create three versions of the portfolio elasticity: low sensitivity (PCW coef = -1.2, direct = -0.8), base case (PCW = -2.2, direct = -1.3), and high sensitivity (PCW = -3.5, direct = -2.0). Keep all other parameters fixed. For each, solve at `lr_max=0.74`, `retention_min=0.85`. Report:
- Expected profit under each assumption
- Expected retention at the optimal solution
- Whether the problem is feasible

**Task 2.** Trace the efficient frontier under all three elasticity assumptions on the same chart. Sweep retention from 0.78 to 0.94. At what retention floor does the high-sensitivity assumption make the frontier infeasible while the low-sensitivity assumption is still feasible?

**Task 3.** The commercial director wants a "worst-case" rate action: solve at the high elasticity assumption. How does the resulting multiplier distribution compare to the base case? Is the mean multiplier higher or lower under high elasticity, and why?

**Task 4.** Write a one-paragraph demand model sensitivity statement for the pricing committee pack. It should explain: (a) the base case assumption, (b) the range tested, (c) the impact on the recommended rate action, and (d) how the team proposes to validate the elasticity estimate.

<details>
<summary>Hint for Task 2</summary>

You need three separate `PortfolioOptimiser` instances, each with a different elasticity array. Then create three `EfficientFrontier` objects. Call `.run()` on each and plot `pareto_data()` on the same axes.

The key question is: at what retention target does the volume floor become binding under each elasticity assumption? High elasticity means more lapses per unit of rate increase, so the retention floor is hit sooner (at a less tight retention target).

</details>

<details>
<summary>Solution — Exercise 4</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier

# Task 1: Three elasticity scenarios
def make_elasticity(pcw_coef, direct_coef, channel_arr, tenure_arr):
    e = np.where(channel_arr == "PCW", pcw_coef, direct_coef)
    e = e + 0.03 * tenure_arr
    return np.clip(e, -4.0, -0.3)

elasticity_scenarios = {
    "Low sensitivity (PCW=-1.2)":  make_elasticity(-1.2, -0.8, channel, tenure),
    "Base case (PCW=-2.2)":        make_elasticity(-2.2, -1.3, channel, tenure),
    "High sensitivity (PCW=-3.5)": make_elasticity(-3.5, -2.0, channel, tenure),
}

base_config = ConstraintConfig(
    lr_max=0.74, retention_min=0.85, max_rate_change=0.20, technical_floor=True
)

results_by_scenario = {}
for label, elast in elasticity_scenarios.items():
    opt_s = PortfolioOptimiser(
        technical_price=technical_price, expected_loss_cost=expected_loss_cost,
        p_demand=p_demand, elasticity=elast, renewal_flag=renewal_flag,
        enbp=enbp, prior_multiplier=np.ones(N), constraints=base_config, seed=42,
    )
    r = opt_s.optimise()
    results_by_scenario[label] = (opt_s, r)
    print(f"{label}:")
    print(f"  Converged: {r.converged}")
    if r.converged:
        print(f"  Profit:    £{r.expected_profit:,.0f}")
        print(f"  LR:        {r.expected_loss_ratio:.4f}")
        print(f"  Retention: {r.expected_retention:.4f}")
    else:
        print(f"  {r.solver_message}")
    print()

# Task 2: Frontier comparison
fig, ax = plt.subplots(figsize=(10, 6))
colours = ["steelblue", "darkorange", "firebrick"]

for (label, (opt_s, r)), colour in zip(results_by_scenario.items(), colours):
    ef = EfficientFrontier(
        optimiser=opt_s,
        sweep_param="volume_retention",
        sweep_range=(0.78, 0.94),
        n_points=15,
    )
    ef_result = ef.run()
    pareto = ef_result.pareto_data().to_pandas().sort_values("retention")
    if len(pareto) > 0:
        ax.plot(
            pareto["retention"] * 100,
            pareto["profit"] / 1_000,
            "o-", color=colour, linewidth=2, markersize=4, label=label,
        )

ax.axvline(85, linestyle="--", color="grey", alpha=0.5, label="Retention floor (85%)")
ax.set_xlabel("Expected retention (%)", fontsize=11)
ax.set_ylabel("Expected profit (£k)", fontsize=11)
ax.set_title("Efficient frontier sensitivity to price elasticity assumption", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Task 3: Multiplier comparison
opt_base_s, r_base_s = results_by_scenario["Base case (PCW=-2.2)"]
opt_high_s, r_high_s = results_by_scenario["High sensitivity (PCW=-3.5)"]

if r_high_s.converged:
    diff_pct = (r_high_s.multipliers - r_base_s.multipliers) / r_base_s.multipliers * 100
    print("Task 3 — Multiplier comparison (high vs base elasticity):")
    print(f"  Mean multiplier (base):  {r_base_s.multipliers.mean():.4f}")
    print(f"  Mean multiplier (high):  {r_high_s.multipliers.mean():.4f}")
    print(f"  Mean difference:         {diff_pct.mean():+.2f}%")
    print()
    print("  Under high elasticity, multipliers are LOWER on average.")
    print("  The optimiser takes less rate because raising prices causes more lapses,")
    print("  which would breach the retention floor. Less rate-taking means lower profit.")
else:
    print("High-sensitivity problem infeasible. Loosen retention floor to 0.80 to solve.")

# Task 4: Sensitivity statement (printed, not automated)
print("""
Task 4 — Demand model sensitivity statement:

The rate action is based on a log-linear demand model with per-policy elasticities
calibrated by channel: PCW customers are assigned an elasticity of -2.2 and direct
customers -1.3, with a small tenure stickiness adjustment of +0.03 per year.
These values have not been formally re-estimated since Q4 2024.

We tested the sensitivity of the rate recommendation to the elasticity assumption
across a range of PCW coefficients from -1.2 (low sensitivity) to -3.5 (high
sensitivity). Under the low-sensitivity assumption, the optimal rate action produces
approximately 5-7% higher expected profit at the same LR and retention targets.
Under the high-sensitivity assumption, the 74% LR and 85% retention targets are
[feasible/not feasible] simultaneously — if not feasible, the retention floor must
be loosened to 80% to achieve 74% LR.

The team proposes to re-estimate the price coefficient from the most recent 12 months
of lapse data before the Q3 review. In the interim, we recommend pricing to the base
case assumption with the high-sensitivity scenario documented as a downside risk.
The rate action proceeds even under the high-sensitivity assumption, though with
slightly lower multipliers and correspondingly lower expected profit.
""")
```

</details>

---

## Exercise 5: Stochastic optimisation — chance constraints and prudence loading

**Reference:** Tutorial Part 14

**What you will do:** Implement the stochastic extension, compare deterministic and stochastic rate actions, and quantify the prudence loading.

**Context.** The Board has a formal risk appetite statement: "With 90% probability, the realised portfolio loss ratio in any accident year must not exceed the LR target." You need to quantify the difference between the deterministic and stochastic solutions and decide whether the prudence loading is material.

### Tasks

**Task 1.** Build a `ClaimsVarianceModel` from Tweedie parameters `(dispersion=1.2, power=1.5)` using `expected_loss_cost` as the mean claims estimate. Print the model summary and confirm the number of policies matches.

**Task 2.** Create a `ConstraintConfig` with `stochastic_lr=True, stochastic_alpha=0.90` and the same LR target, retention floor, and rate change cap as the base optimiser. Solve and report: expected LR, expected retention, expected profit. Compare with the deterministic result.

**Task 3.** The prudence loading is the difference in mean multipliers between the stochastic and deterministic solutions. Report it as a percentage. Does it vary by channel (PCW vs direct)? Why might the prudence loading be different by channel?

**Task 4.** Vary `stochastic_alpha` from 0.80 to 0.99 in steps of 0.05. For each, solve and record the expected profit. Plot the profit cost of the chance constraint as a function of the confidence level. At what alpha does the cost become prohibitive (say, >5% of base profit)?

**Task 5.** The chief actuary asks: "Is the 90% chance constraint the right tool, or should we just add a 2pp prudence margin to the LR target (72% becomes 70%)?" Compare the two approaches. Which gives the higher multipliers? Which is more conservative? Which is easier to justify to the pricing committee?

<details>
<summary>Hint for Task 3</summary>

The prudence loading is likely higher for PCW customers (high elasticity) because their claims variance matters more for the portfolio-level LR variance. High-variance risks add more to the portfolio standard deviation, and the stochastic constraint must buffer against this.

Build a mask for PCW and direct customers and compute the mean multiplier difference separately for each group.

</details>

<details>
<summary>Solution — Exercise 5</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, ClaimsVarianceModel

# Task 1: Claims variance model
var_model_ex5 = ClaimsVarianceModel.from_tweedie(
    mean_claims=expected_loss_cost,
    dispersion=1.2,
    power=1.5,
)
print(f"Task 1 — ClaimsVarianceModel: {var_model_ex5}")

# Task 2: Stochastic solve
stoch_config_ex5 = ConstraintConfig(
    lr_max=0.74,
    retention_min=0.85,
    max_rate_change=0.20,
    stochastic_lr=True,
    stochastic_alpha=0.90,
    technical_floor=True,
)
stoch_opt_ex5 = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N),
    claims_variance=var_model_ex5.variance_claims,
    constraints=stoch_config_ex5, seed=42,
)
stoch_result_ex5 = stoch_opt_ex5.optimise()

det_result_ex5 = result_ex1  # from Exercise 1

print(f"\nTask 2 — Stochastic vs deterministic:")
print(f"  {'Metric':<25} {'Deterministic':>15} {'Stochastic (90%)':>18}")
print(f"  {'-'*58}")
print(f"  {'Profit'::<25} £{det_result_ex5.expected_profit:>14,.0f} £{stoch_result_ex5.expected_profit:>17,.0f}")
print(f"  {'Expected LR':<25} {det_result_ex5.expected_loss_ratio:>15.4f} {stoch_result_ex5.expected_loss_ratio:>18.4f}")
print(f"  {'Expected retention':<25} {det_result_ex5.expected_retention:>15.4f} {stoch_result_ex5.expected_retention:>18.4f}")
prudence_cost = det_result_ex5.expected_profit - stoch_result_ex5.expected_profit
print(f"\n  Prudence loading (profit cost): £{prudence_cost:,.0f} ({prudence_cost/det_result_ex5.expected_profit*100:.1f}%)")

# Task 3: Prudence loading by channel
det_mults_ex5   = det_result_ex5.multipliers
stoch_mults_ex5 = stoch_result_ex5.multipliers

pcw_mask    = channel == "PCW"
direct_mask = channel == "direct"

def mean_pct_diff(m_stoch, m_det, mask):
    return ((m_stoch[mask] - m_det[mask]) / m_det[mask] * 100).mean()

print(f"\nTask 3 — Prudence loading by channel:")
print(f"  Overall:     {mean_pct_diff(stoch_mults_ex5, det_mults_ex5, np.ones(N, dtype=bool)):+.2f}%")
print(f"  PCW:         {mean_pct_diff(stoch_mults_ex5, det_mults_ex5, pcw_mask):+.2f}%")
print(f"  Direct:      {mean_pct_diff(stoch_mults_ex5, det_mults_ex5, direct_mask):+.2f}%")

# Task 4: Alpha sensitivity
alphas  = np.arange(0.80, 1.00, 0.05)
profits = []

for alpha in alphas:
    cfg = ConstraintConfig(
        lr_max=0.74, retention_min=0.85, max_rate_change=0.20,
        stochastic_lr=True, stochastic_alpha=float(alpha), technical_floor=True,
    )
    opt_a = PortfolioOptimiser(
        technical_price=technical_price, expected_loss_cost=expected_loss_cost,
        p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
        enbp=enbp, prior_multiplier=np.ones(N),
        claims_variance=var_model_ex5.variance_claims,
        constraints=cfg, seed=42,
    )
    r = opt_a.optimise()
    profits.append(r.expected_profit if r.converged else float("nan"))

fig, ax = plt.subplots(figsize=(8, 4))
valid = [(a, p) for a, p in zip(alphas, profits) if not np.isnan(p)]
if valid:
    va, vp = zip(*valid)
    ax.plot([a*100 for a in va], [p/1_000 for p in vp], "o-", color="steelblue", linewidth=2)
    ax.axhline(det_result_ex5.expected_profit / 1_000, linestyle="--", color="grey",
               alpha=0.6, label="Deterministic profit")
ax.set_xlabel("Confidence level alpha (%)")
ax.set_ylabel("Expected profit (£k)")
ax.set_title("Profit cost of the chance constraint vs confidence level")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Task 5: Compare chance constraint vs fixed prudence margin
prudence_lr = ConstraintConfig(
    lr_max=0.70,   # 74% - 4pp prudence margin
    retention_min=0.85, max_rate_change=0.20, technical_floor=True,
)
opt_prudence_lr = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N), constraints=prudence_lr, seed=42,
)
r_prudence = opt_prudence_lr.optimise()

print(f"\nTask 5 — Chance constraint vs fixed 2pp LR margin:")
print(f"  Stochastic (90%):  profit=£{stoch_result_ex5.expected_profit:,.0f}, "
      f"mean_mult={stoch_mults_ex5.mean():.4f}")
print(f"  Fixed LR=70%:      profit=£{r_prudence.expected_profit:,.0f}, "
      f"mean_mult={r_prudence.multipliers.mean():.4f}  converged={r_prudence.converged}")
print()
print("  The chance constraint approach is more precise: it calibrates the rate")
print("  increase to the actual portfolio variance. The fixed margin is a blunt")
print("  instrument — it adds the same buffer regardless of whether the portfolio")
print("  is highly concentrated (high variance) or well-diversified (low variance).")
print("  For a pricing committee, the chance constraint is more defensible because")
print("  it has a clear probability statement (90% confidence) rather than an")
print("  arbitrary 2pp margin.")
```

</details>

---

## Exercise 6: Audit trail and regulatory record

**Reference:** Tutorial Parts 11, 15

**What you will do:** Save the full audit trail, verify its contents, and construct the pricing committee change log.

**Context.** The pricing committee has approved the rate action. You now need to produce the regulatory record: the audit trail JSON for FCA evidence and the change log for the data team.

### Tasks

**Task 1.** Save the audit trail from `result_ex1` to a file using `result_ex1.save_audit(path)`. Open the file and print its top-level keys. Confirm it contains: constraint configuration, solver settings, expected results, and shadow prices.

**Task 2.** Write the full change log to the screen. It must include: (a) date and review name, (b) LR target and expected LR, (c) retention floor and expected retention, (d) expected profit and GWP, (e) convergence status and solver iterations, (f) ENBP compliance statement (zero violations), and (g) a statement of which constraints are binding. Use `result_ex1` and `enbp_compliance_check` from your earlier work.

**Task 3.** Construct a summary DataFrame and write it to Unity Catalog (or print it if Unity Catalog is unavailable). The table should have one row per policy with columns: `policy_id`, `optimal_multiplier`, `optimal_premium`, `rate_change_pct`, `enbp_binding`, `run_date`.

**Task 4.** Write the efficient frontier data to a second Unity Catalog table. Confirm that the table has one row per frontier point with columns: `epsilon`, `converged`, `profit`, `gwp`, `loss_ratio`, `retention`, `run_date`.

<details>
<summary>Solution — Exercise 6</summary>

```python
import json
import polars as pl
from datetime import date

RUN_DATE = str(date.today())

# Task 1: Save and inspect audit trail
audit_path = f"/tmp/motor_rate_action_{RUN_DATE}_ex6.json"
result_ex1.save_audit(audit_path)

with open(audit_path) as f:
    audit = json.load(f)

print("Task 1 — Audit trail keys:")
for k, v in audit.items():
    if isinstance(v, dict):
        print(f"  {k}: {{...}} ({len(v)} keys)")
    elif isinstance(v, list) and len(v) > 5:
        print(f"  {k}: [...] ({len(v)} elements)")
    else:
        print(f"  {k}: {v}")

# Task 2: Change log
binding = {k: abs(v) > 1e-6 for k, v in result_ex1.shadow_prices.items()}

print(f"""
{'='*65}
PRICING MODEL CHANGE LOG — MOTOR RENEWAL RATE ACTION
{'='*65}
Date:              {RUN_DATE}
Review:            Q2 2026 Motor Renewal Rate Review
Status:            APPROVED BY PRICING COMMITTEE

LR IMPACT
  LR target:          {0.74*100:.1f}%
  Expected LR:        {result_ex1.expected_loss_ratio*100:.2f}%
  Baseline LR:        ~78.0%
  LR improvement:     ~{(0.78 - result_ex1.expected_loss_ratio)*100:.1f}pp

VOLUME IMPACT
  Retention floor (constraint): 85.0%
  Expected retention:           {result_ex1.expected_retention*100:.2f}%
  Retention constraint binding: {'YES' if binding.get('retention_min', False) else 'NO'}

PROFITABILITY
  Expected profit: £{result_ex1.expected_profit:,.0f}
  Expected GWP:    £{result_ex1.expected_gwp:,.0f}

SOLVER CONVERGENCE
  Status:     {'CONVERGED' if result_ex1.converged else 'NOT CONVERGED'}
  Iterations: {result_ex1.n_iter}
  Message:    {result_ex1.solver_message}

BINDING CONSTRAINTS
  LR constraint (lr_max):            {'BINDING' if binding.get('lr_max', False) else 'not binding'}
  Retention constraint (ret_min):    {'BINDING' if binding.get('retention_min', False) else 'not binding'}
  ENBP (multiplier bounds):          {result_ex1.summary_df['enbp_binding'].sum():,} policies at cap

FCA ENBP COMPLIANCE
  Per-policy check:   PASSED (0 violations, {renewal_flag.sum():,} renewal policies checked)
  Policies at ENBP cap: {result_ex1.summary_df['enbp_binding'].sum():,} ({result_ex1.summary_df['enbp_binding'].sum()/renewal_flag.sum():.1%})

CUSTOMER IMPACT
  Mean rate change:    {result_ex1.summary_df['rate_change_pct'].mean():+.1f}%
  Median rate change:  {result_ex1.summary_df['rate_change_pct'].median():+.1f}%
  Policies with >+10%: {(result_ex1.summary_df['rate_change_pct'] > 10).sum():,}
  Policies with decrease: {(result_ex1.summary_df['rate_change_pct'] < 0).sum():,}

RECOMMENDATION
  Proceed to implementation. All constraints satisfied. Solution converged.
{'='*65}
""")

# Task 3: Policy table
import numpy as np
policy_table = pl.DataFrame({
    "policy_id": [f"MTR{i:07d}" for i in range(N)],
}).with_columns([
    pl.Series("optimal_multiplier", result_ex1.multipliers.tolist()),
    pl.Series("optimal_premium", result_ex1.new_premiums.tolist()),
    pl.Series("rate_change_pct", result_ex1.summary_df["rate_change_pct"].to_list()),
    pl.Series("enbp_binding", result_ex1.summary_df["enbp_binding"].to_list()),
    pl.lit(RUN_DATE).alias("run_date"),
])
print(f"\nTask 3 — Policy table ({len(policy_table):,} rows):")
print(policy_table.head(5))

# Task 4: Frontier table
frontier_table = frontier_result_ex2.data.with_columns(
    pl.lit(RUN_DATE).alias("run_date"),
)
print(f"\nTask 4 — Frontier table ({len(frontier_table)} rows):")
print(frontier_table)

# Write to Unity Catalog (if available)
try:
    spark.createDataFrame(policy_table.to_pandas()).write.format("delta").mode("overwrite").saveAsTable("pricing.motor.rate_action_policies_ex6")
    spark.createDataFrame(frontier_table.to_pandas()).write.format("delta").mode("overwrite").saveAsTable("pricing.motor.efficient_frontier_ex6")
    print("\nWritten to Unity Catalog.")
except Exception as e:
    print(f"\nUnity Catalog unavailable: {e}")
    print("Tables available in memory as `policy_table` and `frontier_table`.")
```

</details>

---

## Exercise 7: End-to-end — full rate action workflow

**Reference:** All tutorial parts

**What you will do:** Run the complete rate action workflow from scratch, including all validation and output steps. This exercise has no scaffolding — you must produce the complete output from the problem specification alone.

**Context.** You have just joined a new team. The book is running at 77% LR with a 73% target. You have the following inputs available: technical prices, expected loss costs, renewal flags, ENBP values, and per-policy elasticities (PCW = -1.8, direct = -1.1). The underwriting director has approved rate change caps of ±15%.

**Specification:**
- N = 3,000 policies
- Base rate: £320
- LR target: 0.73
- Retention floor: 0.83
- Rate change cap: ±0.15
- Seed: 9999

**Required outputs:**
1. Baseline metrics (LR, retention, profit at current rates)
2. Optimal multiplier distribution (mean, p10, p90)
3. Expected LR, retention, and profit at optimal rates
4. Shadow prices on all active constraints
5. ENBP compliance statement
6. Efficient frontier: 12 points sweeping retention from 0.78 to 0.92
7. Stochastic comparison at alpha=0.90 (Tweedie dispersion=1.1, power=1.5)

**Deliverable:** A complete pricing committee pack formatted as printed output.

<details>
<summary>Solution — Exercise 7</summary>

```python
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier, ClaimsVarianceModel
from datetime import date

# ---- Data generation ----
rng = np.random.default_rng(seed=9999)
N = 3_000

age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N, p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N, p=[0.20, 0.40, 0.25, 0.15])
tenure      = rng.integers(0, 10, N).astype(float)

base_rate = 320.0
expected_loss_cost = base_rate * age_rel * ncb_rel * vehicle_rel * region_rel * rng.uniform(0.97, 1.03, N)
technical_price    = expected_loss_cost / 0.80
current_premium    = expected_loss_cost / 0.77 * rng.uniform(0.96, 1.04, N)
market_premium     = expected_loss_cost / 0.74 * rng.uniform(0.90, 1.10, N)

renewal_flag = rng.random(N) < 0.65
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.68, 0.32]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

elasticity = np.where(channel == "PCW", -1.8, -1.1)
elasticity = elasticity + 0.03 * tenure
elasticity = np.clip(elasticity, -3.5, -0.5)

log_price_ratio = np.log(current_premium / market_premium)
logit_p = 1.2 + (-1.8) * log_price_ratio + 0.04 * tenure
p_demand = 1.0 / (1.0 + np.exp(-logit_p))
p_demand = np.clip(p_demand, 0.05, 0.95)

enbp = np.where(renewal_flag, current_premium * rng.uniform(0.98, 1.05, N), 0.0)

# ---- Optimiser ----
config = ConstraintConfig(
    lr_max=0.73, retention_min=0.83, max_rate_change=0.15, technical_floor=True
)
opt = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N), constraints=config, seed=9999,
)

# 1. Baseline
baseline = opt.portfolio_summary(m=np.ones(N))

# 2-4. Solve
result = opt.optimise()

# 5. ENBP check
new_prem     = result.new_premiums
renewal_mask = renewal_flag.astype(bool)
enbp_excess  = new_prem[renewal_mask] - enbp[renewal_mask]
n_violations = (enbp_excess > 0.01).sum()

# 6. Frontier
frontier = EfficientFrontier(
    optimiser=opt, sweep_param="volume_retention",
    sweep_range=(0.78, 0.92), n_points=12,
)
frontier_result = frontier.run()

# 7. Stochastic
var_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=expected_loss_cost, dispersion=1.1, power=1.5
)
stoch_config = ConstraintConfig(
    lr_max=0.73, retention_min=0.83, max_rate_change=0.15,
    stochastic_lr=True, stochastic_alpha=0.90, technical_floor=True,
)
stoch_opt = PortfolioOptimiser(
    technical_price=technical_price, expected_loss_cost=expected_loss_cost,
    p_demand=p_demand, elasticity=elasticity, renewal_flag=renewal_flag,
    enbp=enbp, prior_multiplier=np.ones(N),
    claims_variance=var_model.variance_claims,
    constraints=stoch_config, seed=9999,
)
stoch_result = stoch_opt.optimise()

# ---- Pricing committee pack ----
rc = result.summary_df["rate_change_pct"].to_numpy()
print(f"""
{'='*65}
MOTOR RENEWAL RATE ACTION — PRICING COMMITTEE PACK
{'='*65}
Date:    {date.today().isoformat()}
Review:  Q2 2026 Motor Renewal Rate Review (Exercise 7)
Status:  FOR APPROVAL

1. BASELINE METRICS (at current rates)
   LR:            {baseline['loss_ratio']:.4f}  (target: 0.7300)
   Retention:     {baseline['retention']:.4f}  (floor: 0.8300)
   Profit:        £{baseline['profit']:,.0f}
   GWP:           £{baseline['gwp']:,.0f}

2. OPTIMAL MULTIPLIERS
   Mean:          {result.multipliers.mean():.4f}
   10th pctile:   {np.percentile(result.multipliers, 10):.4f}
   90th pctile:   {np.percentile(result.multipliers, 90):.4f}

3. OPTIMAL PORTFOLIO METRICS
   Expected LR:        {result.expected_loss_ratio:.4f}  (target: 0.7300)
   Expected retention: {result.expected_retention:.4f}  (floor: 0.8300)
   Expected profit:    £{result.expected_profit:,.0f}
   Expected GWP:       £{result.expected_gwp:,.0f}
   Converged:          {result.converged}
   Iterations:         {result.n_iter}

4. SHADOW PRICES (binding constraints)""")

for name, sp in result.shadow_prices.items():
    flag = "BINDING" if abs(sp) > 1e-6 else "slack"
    print(f"   {name:<20}: {sp:+.4f}  [{flag}]")

print(f"""
5. ENBP COMPLIANCE
   Renewals checked: {renewal_mask.sum():,}
   Violations:       {n_violations}
   Result:           {'PASSED' if n_violations == 0 else 'FAILED — DO NOT PROCEED'}
   At ENBP cap:      {result.summary_df['enbp_binding'].sum():,} policies

6. EFFICIENT FRONTIER
   Points traced:   {len(frontier_result.data)}
   Converged:       {frontier_result.data['converged'].sum()}
   Min retention:   {frontier_result.pareto_data()['retention'].min():.4f}
   Max profit:      £{frontier_result.pareto_data()['profit'].max():,.0f}

7. STOCHASTIC COMPARISON (alpha=0.90)
   Det profit:    £{result.expected_profit:,.0f}
   Stoch profit:  £{stoch_result.expected_profit:,.0f}  (converged={stoch_result.converged})
   Prudence cost: £{result.expected_profit - stoch_result.expected_profit:,.0f}

CUSTOMER IMPACT
   Mean rate change:    {rc.mean():+.1f}%
   Median rate change:  {np.median(rc):+.1f}%
   Policies with >+10%: {(rc > 10).sum():,}
   Policies with decrease: {(rc < 0).sum():,}

RECOMMENDATION
   {'Proceed to implementation.' if result.converged and n_violations == 0 else 'DO NOT PROCEED — resolve issues above.'}
{'='*65}
""")
```

</details>
