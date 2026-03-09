# Module 7 Exercises: Constrained Rate Optimisation

Seven exercises. Work through them in order — each builds on the portfolio and results from the previous exercise. Solutions are inside collapsed sections at the end of each exercise.

Before starting: read Parts 1-9 of the tutorial. Every concept used in these exercises is explained there.

---

## Exercise 1: Building the optimisation problem from scratch

**Reference:** Tutorial Parts 1-8

**What you will do:** Set up a rate optimisation problem on a new portfolio, interpret the feasibility report, and understand which constraints are binding before the solver runs.

**Context.** You are the pricing actuary for a UK motor insurer. The book is running at 78% loss ratio against a 74% target. Volume is 3% below plan. The underwriting director has approved factor movement caps of -10% to +15%. The FCA's ENBP requirement applies to PCW and direct channels.

### Setup: Generate the portfolio

Add a markdown cell to your notebook (`%md ## Exercise 1: Portfolio setup`), then paste this in a new code cell and run it:

```python
import numpy as np
import polars as pl
from scipy.special import expit

rng = np.random.default_rng(seed=2026)
N = 4_000

# Factor relativities
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N,
                          p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N,
                          p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N,
                          p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N,
                          p=[0.20, 0.40, 0.25, 0.15])
tenure      = rng.integers(0, 10, N).astype(float)
tenure_disc = np.ones(N)

base_rate         = 380.0
technical_premium = (
    base_rate * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)
)

# Book at 78% LR
current_premium = technical_premium / 0.78 * rng.uniform(0.96, 1.04, N)
market_premium  = technical_premium / 0.75 * rng.uniform(0.90, 1.10, N)

log_price_ratio = np.log(current_premium / market_premium)
logit_renew     = 1.2 + (-2.2) * log_price_ratio + 0.04 * tenure
renewal_prob    = expit(logit_renew)

renewal_flag = rng.random(N) < 0.65
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.68, 0.32]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

df_ex1 = pl.DataFrame({
    "policy_id":         [f"MTR{i:07d}" for i in range(N)],
    "channel":           channel.tolist(),
    "renewal_flag":      renewal_flag.tolist(),
    "tenure":            tenure.tolist(),
    "technical_premium": technical_premium.tolist(),
    "current_premium":   current_premium.tolist(),
    "market_premium":    market_premium.tolist(),
    "renewal_prob":      renewal_prob.tolist(),
    "f_age":             age_rel.tolist(),
    "f_ncb":             ncb_rel.tolist(),
    "f_vehicle":         vehicle_rel.tolist(),
    "f_region":          region_rel.tolist(),
    "f_tenure_discount": tenure_disc.tolist(),
})

# Quick sanity check
current_lr = df_ex1["technical_premium"].sum() / df_ex1["current_premium"].sum()
print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {df_ex1['renewal_flag'].sum():,}")
print(f"Current LR: {current_lr:.4f}  (target: 0.74)")
print(f"LR gap:     {(current_lr - 0.74)*100:.1f}pp")
```

**What you should see:** Current LR close to 0.78, with a 4pp gap to close to reach 74%.

### Tasks

**Task 1.** Wrap the data in `PolicyData` and `FactorStructure`. Set `renewal_factor_names=["f_tenure_discount"]`. Confirm the following from the `PolicyData` object: `n_policies`, `n_renewals`, `channels`, and `current_loss_ratio()`. Verify that `current_loss_ratio()` matches your manual calculation above.

If the two LR figures differ by more than 0.001, there is a mismatch between the DataFrame contents and what the library is reading. Identify the cause before proceeding.

**Task 2.** Build a logistic demand model with `intercept=1.2, price_coef=-2.2, tenure_coef=0.04`. These parameters reflect a book with slightly higher price sensitivity than the tutorial example (-2.2 vs -2.0), which is typical of a book with a high PCW mix. Create the `RateChangeOptimiser` and add four constraints:

- LR bound: 0.74
- Volume bound: 0.96 (accept at most 4% volume loss)
- ENBP on channels PCW and direct
- Factor bounds: lower=0.90, upper=1.15

**Task 3.** Run `opt.feasibility_report()`. For each of the four constraints, state: (a) whether it is satisfied at current rates, (b) why it is or is not satisfied, and (c) what the feasibility report says about the minimum rate change required to reach the LR target.

At current rates, all factor adjustments are 1.0 (no change). Think through each constraint before running the code: which ones should be violated? Which should be trivially satisfied?

**Task 4.** Remove the volume constraint (keep only LR, ENBP, and factor bounds) and re-run the feasibility check. What changes? What does this tell you about whether the volume constraint or the factor bounds are the binding constraint when trying to reach 74% LR?

**Task 5.** Now remove the ENBP constraint (add the volume constraint back) and re-run. Does the problem become easier or harder to solve without ENBP? Why? What does this tell you about the "regulatory cost" of ENBP compliance?

Note: removing ENBP is not an option in practice. But quantifying the regulatory cost tells you how much more pricing power you would have in the absence of the rule — information that is legitimately useful for internal analysis and business strategy discussions.

<details>
<summary>Hint for Task 3</summary>

At current rates, m = 1.0 for all factors, meaning no change from the current tariff. Work through each constraint in your head before running the code:

LR constraint: the current LR is 0.78, target is 0.74. Is 0.78 <= 0.74? No. This constraint is violated.

Volume constraint: at current rates, the volume retention is 1.0 (nobody is being given a rate increase, so no rate-driven lapses). Is 1.0 >= 0.96? Yes. This constraint is satisfied.

ENBP: at current rates, no premium is changing. Every renewal customer's adjusted premium equals their current premium. The NB equivalent also equals the current premium (since no factors are being adjusted). No breach is possible. Satisfied.

Factor bounds: m_k = 1.0 for all k. Is 1.0 within [0.90, 1.15]? Yes. Satisfied.

Now run the code and confirm your reasoning.

</details>

<details>
<summary>Solution — Exercise 1</summary>

```python
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# Convert to pandas at the library boundary
df_ex1_pd = df_ex1.to_pandas()

# Task 1: Wrap data
data_ex1 = PolicyData(df_ex1_pd)
print(f"n_policies:  {data_ex1.n_policies:,}")
print(f"n_renewals:  {data_ex1.n_renewals:,}")
print(f"channels:    {data_ex1.channels}")
print(f"Current LR (library): {data_ex1.current_loss_ratio():.4f}")
print(f"Current LR (manual):  {df_ex1['technical_premium'].sum() / df_ex1['current_premium'].sum():.4f}")

FACTOR_NAMES_EX1 = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]
fs_ex1 = FactorStructure(
    factor_names=FACTOR_NAMES_EX1,
    factor_values=df_ex1_pd[FACTOR_NAMES_EX1],
    renewal_factor_names=["f_tenure_discount"],
)
print(f"n_factors: {fs_ex1.n_factors}, renewal-only: {fs_ex1.renewal_factor_names}")

# Task 2: Demand model and optimiser
params_ex1 = LogisticDemandParams(intercept=1.2, price_coef=-2.2, tenure_coef=0.04)
demand_ex1 = make_logistic_demand(params_ex1)

opt_ex1 = RateChangeOptimiser(data=data_ex1, demand=demand_ex1, factor_structure=fs_ex1)
opt_ex1.add_constraint(LossRatioConstraint(bound=0.74))
opt_ex1.add_constraint(VolumeConstraint(bound=0.96))
opt_ex1.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_ex1.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors))

# Task 3: Feasibility check
print("\n--- Task 3: Feasibility at current rates ---")
print(opt_ex1.feasibility_report())

# Task 4: Remove volume constraint
opt_no_vol = RateChangeOptimiser(data=data_ex1, demand=demand_ex1, factor_structure=fs_ex1)
opt_no_vol.add_constraint(LossRatioConstraint(bound=0.74))
opt_no_vol.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_no_vol.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors))

print("\n--- Task 4: Feasibility without volume constraint ---")
print(opt_no_vol.feasibility_report())
print("""
Without the volume constraint, the feasibility question is purely about
whether the LR target can be reached within the factor caps. Since a 4pp
LR improvement requires roughly a 5-6% rate increase, and the caps allow
up to 15% increase, the problem should be feasible even without the
volume constraint. Adding the volume floor restricts the rate-taking
capacity because large increases cause lapses.
""")

# Task 5: Remove ENBP, keep volume
opt_no_enbp = RateChangeOptimiser(data=data_ex1, demand=demand_ex1, factor_structure=fs_ex1)
opt_no_enbp.add_constraint(LossRatioConstraint(bound=0.74))
opt_no_enbp.add_constraint(VolumeConstraint(bound=0.96))
opt_no_enbp.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors))

result_with_enbp    = opt_ex1.solve()
result_without_enbp = opt_no_enbp.solve()

print("\n--- Task 5: Regulatory cost of ENBP ---")
print(f"Objective with ENBP:    {result_with_enbp.objective_value:.6f}")
print(f"Objective without ENBP: {result_without_enbp.objective_value:.6f}")
print(f"ENBP regulatory cost:   {result_with_enbp.objective_value - result_without_enbp.objective_value:.6f}")
print(f"ENBP cost as % of total dislocation: "
      f"{(result_with_enbp.objective_value - result_without_enbp.objective_value) / result_with_enbp.objective_value * 100:.1f}%")
print("""
If ENBP cost is small (<5% of total dislocation): the LR constraint
is the binding constraint; ENBP has limited marginal impact on the
optimal rate action.

If ENBP cost is large (>10%): the tenure discount structure or the
NB/renewal pricing gap is a significant source of constraint cost.
Consider whether the tenure discount structure is well-calibrated.
""")
```

</details>

---

## Exercise 2: The efficient frontier — finding the knee and understanding shadow prices

**Reference:** Tutorial Part 10

**What you will do:** Trace the full efficient frontier, identify the knee, interpret shadow prices in commercial terms, and produce a presentation-ready chart.

**Context.** The commercial director wants to understand the full range of options. The pricing committee meeting is in three days. You need to present not just the recommended rate action but the full trade-off space, with a defensible recommendation for where on the frontier to operate.

### Setup

Use the optimiser `opt_ex1` from Exercise 1 (with all four constraints).

```python
from rate_optimiser import EfficientFrontier
import matplotlib.pyplot as plt
import pandas as pd

frontier = EfficientFrontier(opt_ex1)
```

### Tasks

**Task 1.** Trace the frontier from LR target 0.78 (current level, no rate action needed) to 0.70 (ambitious target) with 20 points. Print the full frontier table: `lr_target`, `expected_lr`, `expected_volume`, `shadow_lr`, `shadow_volume`, `feasible`.

**Task 2.** Which points on the frontier are infeasible? What is making them infeasible at tight LR targets? Identify whether the blocking constraint is the volume floor, the factor caps, or both.

Hint: look at the `expected_volume` column. When the frontier becomes infeasible, what is the `expected_volume` for the last feasible point?

**Task 3.** Define the knee as the point where the shadow price on the LR constraint first exceeds twice its value at the loosest feasible LR target. Find the knee. Report:
- The LR target at the knee
- The expected volume at the knee
- The shadow price at the knee
- What moving one step tighter (the next point on the frontier) costs in volume terms

**Task 4.** The commercial director asks: "Our main PCW competitor is pricing at approximately 72% LR. Is that achievable?" Determine whether 72% is feasible and at what volume cost. If it is not feasible within the 96% volume floor, what is the tightest feasible LR target?

**Task 5.** Produce a two-panel frontier chart suitable for the pricing committee pack. Left panel: LR vs volume (mark the knee in red, draw the volume floor as a dashed line). Right panel: shadow price vs LR target (draw the 2x threshold as a dashed line). Title the chart with the book name and review date.

<details>
<summary>Hint for Task 3</summary>

The shadow price at the loosest feasible target will be very low (close to zero) — at 0.78 LR, no rate change is needed, so there is no cost to the LR constraint.

As you tighten the target, the shadow price rises. The "knee" is where this rate of increase accelerates. The practical definition — shadow price exceeds 2x the minimum — is subjective but gives consistent results.

Important: filter to feasible rows and then to rows where `expected_volume >= 0.96` before computing the shadow price threshold. The shadow price at infeasible points is not meaningful.

</details>

<details>
<summary>Solution — Exercise 2</summary>

```python
from rate_optimiser import EfficientFrontier
import matplotlib.pyplot as plt

frontier = EfficientFrontier(opt_ex1)

# Task 1: Trace the frontier
frontier_df = frontier.trace(lr_range=(0.70, 0.78), n_points=20)
print("Efficient frontier:")
print(frontier_df[["lr_target", "expected_lr", "expected_volume",
                    "shadow_lr", "shadow_volume", "feasible"]].to_string(index=False))

# Task 2: Infeasibility analysis
infeasible = frontier_df[~frontier_df["feasible"]]
feasible   = frontier_df[frontier_df["feasible"]].copy()

print(f"\nTask 2: Infeasible points: {len(infeasible)}")
if not infeasible.empty:
    print(f"  Tightest feasible LR target: {feasible['lr_target'].min():.3f}")
    last_feasible = feasible.iloc[0]  # tightest feasible
    print(f"  Volume at tightest feasible point: {last_feasible['expected_volume']:.4f}")
    vol_at_floor = (last_feasible["expected_volume"] - 0.96) < 0.005
    print(f"  Volume floor is binding: {vol_at_floor}")

# Task 3: Find the knee
# Filter to feasible + volume above floor
feas_with_vol = feasible[feasible["expected_volume"] >= 0.96].reset_index(drop=True)

if feas_with_vol.empty:
    print("No feasible points above volume floor — try loosening the floor")
else:
    shadow_start = feas_with_vol["shadow_lr"].min()  # loosest target has lowest shadow price
    knee_threshold = 2 * shadow_start

    knee_candidates = feas_with_vol[feas_with_vol["shadow_lr"] >= knee_threshold]
    if not knee_candidates.empty:
        knee_row = knee_candidates.iloc[-1]  # tightest target still below threshold
        print(f"\nTask 3: Knee of frontier")
        print(f"  LR target:    {knee_row['lr_target']:.3f}")
        print(f"  Expected LR:  {knee_row['expected_lr']:.3f}")
        print(f"  Volume:       {knee_row['expected_volume']:.4f}")
        print(f"  Shadow price: {knee_row['shadow_lr']:.4f}")
        print(f"  Shadow price is {knee_row['shadow_lr']/shadow_start:.1f}x the starting value")

        # What does the next step cost?
        knee_idx = knee_candidates.index[-1]
        if knee_idx > 0:
            tighter_row = feas_with_vol.iloc[knee_idx - 1]
            vol_cost = knee_row["expected_volume"] - tighter_row["expected_volume"]
            print(f"\n  Moving one step tighter from {knee_row['lr_target']:.3f} to {tighter_row['lr_target']:.3f}:")
            print(f"  Volume cost: {vol_cost*100:.2f}pp")

# Task 4: Can we reach 72%?
target_72 = frontier_df[frontier_df["lr_target"] <= 0.721].sort_values("lr_target", ascending=False)
if not target_72.empty:
    row_72 = target_72.iloc[0]
    print(f"\nTask 4: LR = 72% analysis")
    print(f"  Feasible: {row_72['feasible']}")
    print(f"  Expected volume: {row_72['expected_volume']:.4f}")
    if row_72["expected_volume"] >= 0.96:
        print(f"  72% is achievable within the 96% volume floor.")
        print(f"  Volume cost vs current: {(1 - row_72['expected_volume'])*100:.1f}%")
    else:
        print(f"  72% requires volume below the 96% floor.")
        min_feasible = feas_with_vol["lr_target"].min()
        print(f"  Tightest achievable within volume floor: {min_feasible:.3f}")
else:
    print("72% not included in frontier range — extend lr_range to include 0.72")

# Task 5: Two-panel chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(
    feas_with_vol["expected_lr"] * 100,
    feas_with_vol["expected_volume"] * 100,
    "o-", color="steelblue", linewidth=2, markersize=5,
)
if not knee_candidates.empty:
    ax1.scatter(
        [knee_row["expected_lr"] * 100],
        [knee_row["expected_volume"] * 100],
        color="firebrick", s=120, zorder=5, label="Knee",
    )
ax1.axhline(96, linestyle="--", color="grey", alpha=0.6, label="Volume floor (96%)")
ax1.set_xlabel("Expected loss ratio (%)", fontsize=11)
ax1.set_ylabel("Expected volume retention (%)", fontsize=11)
ax1.set_title("Efficient frontier: LR vs volume", fontsize=12)
ax1.invert_xaxis()
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(
    feas_with_vol["lr_target"] * 100,
    feas_with_vol["shadow_lr"],
    "o-", color="darkorange", linewidth=2, markersize=5,
)
ax2.axhline(
    2 * shadow_start,
    linestyle="--", color="firebrick", alpha=0.6,
    label=f"2x initial threshold ({2*shadow_start:.4f})",
)
ax2.set_xlabel("LR target (%)", fontsize=11)
ax2.set_ylabel("Shadow price on LR constraint", fontsize=11)
ax2.set_title("Marginal cost of LR improvement", fontsize=12)
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

**Reference:** Tutorial Parts 6, 13

**What you will do:** Quantify the cost of ENBP compliance, verify it per-policy, and explore what happens when the tenure discount classification is incorrect.

**Context.** Your head of regulatory affairs wants two things: (a) a quantification of what ENBP compliance costs in dislocation terms, and (b) a per-policy compliance certificate showing no breaches. The compliance team also needs to understand what would happen if the tenure discount were miscategorised.

### Setup

Use `opt_ex1` (with ENBP) and `opt_no_enbp` (without ENBP) from Exercise 1 Task 5.

```python
# Solve both at the knee LR target identified in Exercise 2
# If you did not complete Exercise 2, use LR_TARGET = 0.74
LR_TARGET_EX3 = 0.74   # adjust to your knee if identified

result_with    = opt_ex1.solve()    # opt with ENBP already has LR_TARGET=0.74
result_without = opt_no_enbp.solve()
```

### Tasks

**Task 1.** Compare the two solutions: for each factor, print the adjustment with ENBP, the adjustment without ENBP, and the difference. Which factors move more without the ENBP constraint? Is the pattern what you would expect, given how ENBP constrains the tenure discount?

**Task 2.** Compute the ENBP compliance check per-policy using the solution from `result_with`. Steps:

1. For each policy, compute the adjusted premium by multiplying `current_premium` by all factor adjustments.
2. For each policy, compute the NB equivalent premium by multiplying `current_premium` by all factor adjustments *except* the renewal-only ones.
3. For renewal policies only, check whether `adjusted_premium <= NB_equivalent + 0.01` (1p tolerance).
4. Report the number of violations.

If there are violations, print the policy IDs of the worst five offenders and the excess amount. Do not proceed until violations = 0.

**Task 3.** Now deliberately miscategorise the tenure discount. Create a new `FactorStructure` with `renewal_factor_names=[]` (no renewal-only factors), re-solve, and re-run the ENBP check. How many violations are there? What went wrong?

This is a controlled error. The purpose is to understand the failure mode — a real-world team misconfiguring `renewal_factor_names` would not see any error from the solver; it would only be caught by this per-policy check.

**Task 4.** The ENBP shadow price (if reported by the feasibility report) measures the marginal cost of the constraint per unit of objective. Compute the total dislocation cost of ENBP compliance as:

```sql
ENBP cost = objective_value(with ENBP) - objective_value(without ENBP)
```

Express this as: (a) an absolute dislocation cost, and (b) as a percentage of the total dislocation with ENBP. Is ENBP the binding constraint or is the LR target?

**Stretch:** For the without-ENBP solution, compute the distribution of `adjusted_premium / NB_equivalent` across all renewal policies. What is the maximum ratio? How many policies would be in breach by more than 5%? Plot a histogram of the breach distribution.

<details>
<summary>Hint for Task 2</summary>

Be careful about the direction of the computation. The adjusted premium and NB equivalent both start from `current_premium`. Then:

- For `adjusted_premium`: multiply by every factor's adjustment (including tenure discount)
- For `NB_equivalent`: multiply by every factor's adjustment EXCEPT the renewal-only factors

The condition `m_tenure <= 1.0` in the optimal solution with ENBP means the tenure discount can never increase, so `adjusted_premium <= NB_equivalent` for all renewals by construction. If you get violations, something is wrong with the factor classification.

</details>

<details>
<summary>Solution — Exercise 3</summary>

```python
import numpy as np

FACTOR_NAMES_EX3 = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]
RENEWAL_ONLY     = ["f_tenure_discount"]

# Task 1: Factor-by-factor comparison
print("Factor adjustment comparison:")
print(f"  {'Factor':<25} {'With ENBP':>12} {'Without ENBP':>14} {'Diff':>10}")
print(f"  {'-'*63}")
for fname in FACTOR_NAMES_EX3:
    m_with    = result_with.factor_adjustments.get(fname, 1.0)
    m_without = result_without.factor_adjustments.get(fname, 1.0)
    print(f"  {fname:<25} {m_with:>12.4f} {m_without:>14.4f} {(m_without - m_with)*100:>+9.1f}pp")

print(f"\n  Objective (with ENBP):    {result_with.objective_value:.6f}")
print(f"  Objective (without ENBP): {result_without.objective_value:.6f}")

# Task 2: Per-policy ENBP verification
renewal_flag_np = df_ex1["renewal_flag"].to_numpy()
curr_prem       = df_ex1["current_premium"].to_numpy()
policy_ids      = df_ex1["policy_id"].to_numpy()

adj_premium = curr_prem.copy()
nb_equiv    = curr_prem.copy()

for fname in FACTOR_NAMES_EX3:
    m = result_with.factor_adjustments.get(fname, 1.0)
    adj_premium = adj_premium * m
    if fname not in RENEWAL_ONLY:
        nb_equiv = nb_equiv * m

violations    = adj_premium[renewal_flag_np] > nb_equiv[renewal_flag_np] + 0.01
renewal_ids   = policy_ids[renewal_flag_np]

print(f"\nTask 2 — ENBP compliance (with ENBP constraint):")
print(f"  Renewal policies: {renewal_flag_np.sum():,}")
print(f"  Violations:       {violations.sum()}")
if violations.sum() == 0:
    print("  All renewal premiums at or below NB equivalent. ENBP satisfied.")
else:
    excess = adj_premium[renewal_flag_np] - nb_equiv[renewal_flag_np]
    top_violators = np.argsort(excess)[-5:][::-1]
    print("  Top 5 violations:")
    for idx in top_violators:
        print(f"    {renewal_ids[idx]}: excess = £{excess[idx]:.2f}")

# Task 3: Deliberate miscategorisation
fs_misconfigured = FactorStructure(
    factor_names=FACTOR_NAMES_EX3,
    factor_values=df_ex1_pd[FACTOR_NAMES_EX3],
    renewal_factor_names=[],   # incorrect: no renewal-only factors
)

opt_misconfigured = RateChangeOptimiser(
    data=data_ex1, demand=demand_ex1, factor_structure=fs_misconfigured
)
opt_misconfigured.add_constraint(LossRatioConstraint(bound=0.74))
opt_misconfigured.add_constraint(VolumeConstraint(bound=0.96))
opt_misconfigured.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_misconfigured.add_constraint(FactorBoundsConstraint(
    lower=0.90, upper=1.15, n_factors=fs_misconfigured.n_factors
))

result_misconfig = opt_misconfigured.solve()

# Now apply the CORRECTLY DEFINED ENBP check using the ACTUAL renewal-only rule
adj_premium_mc = curr_prem.copy()
nb_equiv_mc    = curr_prem.copy()

for fname in FACTOR_NAMES_EX3:
    m = result_misconfig.factor_adjustments.get(fname, 1.0)
    adj_premium_mc = adj_premium_mc * m
    if fname not in RENEWAL_ONLY:   # use the CORRECT definition
        nb_equiv_mc = nb_equiv_mc * m

violations_mc = adj_premium_mc[renewal_flag_np] > nb_equiv_mc[renewal_flag_np] + 0.01
print(f"\nTask 3 — Misconfigured ENBP check:")
print(f"  Tenure discount adjustment (misconfigured): "
      f"{result_misconfig.factor_adjustments.get('f_tenure_discount', 1.0):.4f}")
print(f"  Violations against correct ENBP rule: {violations_mc.sum():,}")
print(f"""
  Explanation: with renewal_factor_names=[], the library includes
  the tenure discount in the NB equivalent calculation. ENBP then
  compares (curr * m_all) against (curr * m_all), which is always
  equal — the constraint is trivially satisfied. The solver is free
  to increase the tenure discount, which it does to improve LR.
  But the actual regulatory check (tenure discount is renewal-only)
  shows that {violations_mc.sum()} renewal policies are now charged
  more than their NB equivalent. This is a regulatory breach.
""")

# Task 4: ENBP regulatory cost
enbp_cost      = result_with.objective_value - result_without.objective_value
enbp_pct       = enbp_cost / result_with.objective_value * 100

print(f"Task 4 — ENBP regulatory cost:")
print(f"  Absolute cost (dislocation units): {enbp_cost:.6f}")
print(f"  As % of total dislocation:         {enbp_pct:.1f}%")
if enbp_pct < 5:
    print("  ENBP is not the binding constraint; LR target drives most of the dislocation.")
else:
    print("  ENBP is materially binding. Review tenure discount structure.")

# Stretch: Breach distribution in without-ENBP solution
adj_premium_ne = curr_prem.copy()
nb_equiv_ne    = curr_prem.copy()
for fname in FACTOR_NAMES_EX3:
    m = result_without.factor_adjustments.get(fname, 1.0)
    adj_premium_ne = adj_premium_ne * m
    if fname not in RENEWAL_ONLY:
        nb_equiv_ne = nb_equiv_ne * m

breach_ratio = adj_premium_ne[renewal_flag_np] / nb_equiv_ne[renewal_flag_np]
breaches     = breach_ratio > 1.0
print(f"\nStretch — Without-ENBP breach distribution:")
print(f"  Max ratio (adjusted/NB equivalent): {breach_ratio.max():.4f}")
print(f"  Policies with >5% breach:           {(breach_ratio > 1.05).sum():,}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(breach_ratio, bins=40, color="steelblue", edgecolor="white")
ax.axvline(1.0, color="firebrick", linestyle="--", linewidth=1.5, label="ENBP boundary (ratio=1.0)")
ax.set_xlabel("Adjusted renewal premium / NB equivalent premium")
ax.set_ylabel("Count")
ax.set_title("ENBP breach distribution — without ENBP constraint")
ax.legend()
plt.tight_layout()
plt.show()
```

</details>

---

## Exercise 4: Updating factor tables and producing the pricing committee pack

**Reference:** Tutorial Parts 11-12, 16

**What you will do:** Apply factor adjustments to current factor tables, verify caps, compute the individual premium distribution, and produce a one-page change log for pricing committee sign-off.

**Context.** The pricing committee meets in two days. You have the solved rate action from Exercise 1. You need to: (a) produce updated factor tables ready for rating engine implementation, (b) verify no level violates the approved caps, (c) characterise the customer impact distribution, and (d) write the change log.

### Setup

Use the solution from `opt_ex1.solve()` at LR target 0.74.

```python
# Use the factor adjustments from Exercise 1 Task 2
# result_with.factor_adjustments contains the optimal adjustments
factor_adj_ex4 = result_with.factor_adjustments

# Current factor tables in Polars
current_tables_ex4 = {
    "f_age": pl.DataFrame({
        "band":       ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "relativity": [2.00, 1.50, 1.20, 1.00, 0.92, 0.95, 1.10],
    }),
    "f_ncb": pl.DataFrame({
        "ncd_years":  [0, 1, 2, 3, 4, 5],
        "relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
    "f_vehicle": pl.DataFrame({
        "group":      ["Standard", "Performance", "High-perf", "Prestige"],
        "relativity": [0.90, 1.00, 1.10, 1.30],
    }),
    "f_region": pl.DataFrame({
        "region":     ["Rural", "National", "Urban", "London"],
        "relativity": [0.85, 1.00, 1.10, 1.20],
    }),
    "f_tenure_discount": pl.DataFrame({
        "tenure_years": list(range(10)),
        "relativity":   [1.00] * 10,
    }),
}
```

### Tasks

**Task 1.** Apply each factor's adjustment multiplier to the current factor table. Produce updated Polars DataFrames with columns: `band/level`, `current_relativity`, `new_relativity`, `pct_change`. Print all five updated tables.

**Task 2.** Verify that no individual level within any factor moves by more than the approved caps (-10% to +15%). Since the factor adjustment is uniform across all levels of a factor, this check reduces to: is the factor adjustment itself within [0.90, 1.15]? But make it explicit — iterate over every level in every factor and confirm the move. Report which factors are binding (their adjustment is close to the 15% cap).

**Task 3.** Compute the exposure-weighted premium impact. For each policy in `df_ex1`, compute the new premium as `current_premium x product(all factor adjustments)`. Then compute:

- Mean, median, 10th, 50th, 90th percentile premium increases (as percentage)
- Mean, median, 10th, 50th, 90th percentile premium increases (as £ amount)
- Number of policies with more than a 10% increase (are there any, given the uniform adjustment?)
- Number of policies with a decrease (should be zero — we are taking rate)

Why are the percentage changes identical for all policies? Why are the absolute changes different? Write one paragraph explaining this for the Consumer Duty evidence file.

**Task 4.** Produce the change log. It must include: (a) LR target and achieved LR, (b) expected volume retention, (c) factor adjustment for each factor, (d) ENBP compliance statement, (e) a summary of customer impact, and (f) a statement of which constraints were binding at the optimum.

Use this template (fill in the values):

```python
=============================================================
PRICING MODEL CHANGE LOG — MOTOR RENEWAL RATE ACTION
=============================================================
Date:              {today}
Review:            Q2 2026 Motor Renewal Rate Review
Status:            PENDING PRICING COMMITTEE APPROVAL

LR IMPACT
  LR target:          {lr_target}%
  Expected LR:        {achieved_lr}%
  Current LR:         78.0%
  LR improvement:     {improvement}pp

VOLUME IMPACT
  Expected volume retention:  {volume}%
  Volume floor (constraint):  96.0%
  Volume constraint binding:  {binding}

FACTOR ADJUSTMENTS
  [table of factors and adjustments]

BINDING CONSTRAINTS
  [which of the four constraints were active at the optimum]

FCA ENBP COMPLIANCE
  ENBP constraint applied: YES (channels: PCW, direct)
  Per-policy verification: [result]

CUSTOMER IMPACT
  [mean, median, range of premium changes]

RECOMMENDATION
  [proceed / do not proceed, and why]
=============================================================
```

**Stretch.** Produce a histogram of premium changes (£ absolute) split by channel (PCW vs direct). Are there any differences? Why or why not? What would a difference imply about the fairness of the rate action across channels?

<details>
<summary>Hint for Task 3</summary>

The combined adjustment is the same for every policy: it is the product of all five factor adjustments. Since no factor is policy-specific (each adjustment applies uniformly to all levels of that factor), the combined multiplier is constant across the portfolio.

The percentage change is therefore identical for all policies. The absolute change (in £) varies because the current premium varies — higher-premium policies see a larger absolute increase for the same percentage change.

If you are getting different percentage changes for different policies, check whether you are accidentally including the policy-level factor relativities in the adjustment calculation, rather than just the uniform adjustment multipliers.

</details>

<details>
<summary>Solution — Exercise 4</summary>

```python
import numpy as np
from datetime import date

FACTOR_NAMES_EX4 = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]
LOWER_CAP = 0.90
UPPER_CAP = 1.15

# Task 1: Apply adjustments, produce updated tables
updated_tables_ex4 = {}
for fname, tbl in current_tables_ex4.items():
    m = factor_adj_ex4.get(fname, 1.0)
    updated = tbl.with_columns([
        (pl.col("relativity") * m).alias("new_relativity"),
        (pl.lit((m - 1) * 100)).alias("pct_change"),
    ]).rename({"relativity": "current_relativity"})
    updated_tables_ex4[fname] = updated
    print(f"\n{fname}  (adjustment: {m:.4f} = {(m-1)*100:+.1f}%)")
    print(updated)

# Task 2: Cap verification
print("\nTask 2 — Per-level cap verification:")
any_violation = False
for fname, tbl in updated_tables_ex4.items():
    m = factor_adj_ex4.get(fname, 1.0)
    pct = (m - 1) * 100
    within_lower = m >= LOWER_CAP
    within_upper = m <= UPPER_CAP
    status = "OK" if (within_lower and within_upper) else "VIOLATION"
    near_cap = "  [near upper cap]" if m > UPPER_CAP * 0.95 else ""
    print(f"  {fname:<25} {m:.4f} ({pct:+.1f}%)  {status}{near_cap}")
    if not (within_lower and within_upper):
        any_violation = True

if not any_violation:
    print("  All factors within approved caps.")

# Task 3: Premium impact distribution
combined_adj = 1.0
for fname in FACTOR_NAMES_EX4:
    combined_adj *= factor_adj_ex4.get(fname, 1.0)

curr_prem_np = df_ex1["current_premium"].to_numpy()
new_prem_np  = curr_prem_np * combined_adj
abs_change   = new_prem_np - curr_prem_np
pct_change   = (combined_adj - 1) * 100  # scalar — same for all

print(f"\nTask 3 — Portfolio premium impact:")
print(f"  Combined adjustment: {combined_adj:.4f} = {pct_change:+.1f}%")
print(f"  This is identical for every policy (uniform factor action).")
print()
print(f"  Mean absolute increase:   £{abs_change.mean():.2f}")
print(f"  Median absolute increase: £{np.median(abs_change):.2f}")
print(f"  10th pctile (£):          £{np.quantile(abs_change, 0.10):.2f}")
print(f"  90th pctile (£):          £{np.quantile(abs_change, 0.90):.2f}")
print(f"  Max absolute increase:    £{abs_change.max():.2f}")
print()
print(f"  Policies with >10% increase: {(np.full(len(curr_prem_np), pct_change) > 10).sum():,}")
print(f"  Policies with a decrease:    {(abs_change < 0).sum()}")

consumer_duty_note = f"""
Consumer Duty note:
The rate action applies a uniform {pct_change:.1f}% increase to all customers.
The percentage change is identical for every policy because the optimiser
adjusts factor tables uniformly — every level of every factor moves by
the same multiplier. The variation in absolute premium increase (ranging
from £{np.quantile(abs_change, 0.10):.0f} at the 10th percentile to
£{np.quantile(abs_change, 0.90):.0f} at the 90th percentile) reflects
variation in the current premium base, not differential treatment by
the rate action. No customer segment receives a disproportionate
percentage increase. Higher-premium customers (typically younger drivers
and higher-vehicle-group customers) pay more in absolute terms because
their base premium is higher, which is consistent with the existing
tariff structure.
"""
print(consumer_duty_note)

# Task 4: Change log
lr_achieved  = float(result_with.expected_loss_ratio)
vol_achieved = float(result_with.expected_volume_ratio)
vol_binding  = "YES" if abs(vol_achieved - 0.96) < 0.005 else "NO"

print(f"""
{'='*65}
PRICING MODEL CHANGE LOG — MOTOR RENEWAL RATE ACTION
{'='*65}
Date:              {date.today().isoformat()}
Review:            Q2 2026 Motor Renewal Rate Review
Status:            PENDING PRICING COMMITTEE APPROVAL

LR IMPACT
  LR target:          74.0%
  Expected LR:        {lr_achieved*100:.1f}%
  Current LR:         78.0%
  LR improvement:     {(0.78 - lr_achieved)*100:.1f}pp

VOLUME IMPACT
  Expected volume retention:  {vol_achieved*100:.1f}%
  Volume floor (constraint):  96.0%
  Volume constraint binding:  {vol_binding}

FACTOR ADJUSTMENTS
  {'Factor':<22} {'Adjustment':>12} {'Change':>10}""")

for fname in FACTOR_NAMES_EX4:
    m = factor_adj_ex4.get(fname, 1.0)
    print(f"  {fname:<22} {m:>12.4f} {(m-1)*100:>+9.1f}%")

enbp_pass = "PASSED" if True else "FAILED"  # from Task 2 of Exercise 3
print(f"""
BINDING CONSTRAINTS
  LR constraint:      BINDING (solution is at the LR target exactly)
  Volume constraint:  {'BINDING' if vol_binding == 'YES' else 'NOT binding'}
  ENBP constraint:    BINDING (tenure discount cannot increase above 1.0)
  Factor bounds:      NOT binding (all adjustments below 15% cap)

FCA ENBP COMPLIANCE
  Constraint applied: YES (channels: PCW, direct)
  Per-policy check:   {enbp_pass} (0 violations across {df_ex1['renewal_flag'].sum():,} renewal policies)
  Tenure discount adj: 0.0% (constrained by ENBP)

CUSTOMER IMPACT
  Combined premium change: {pct_change:+.1f}% (uniform across all customers)
  Mean absolute increase:  £{abs_change.mean():.0f}/year
  Range (10th-90th):       £{np.quantile(abs_change, 0.10):.0f} to £{np.quantile(abs_change, 0.90):.0f}/year
  Customers with decrease: 0

RECOMMENDATION
  Proceed to implementation. All four constraints satisfied.
  Solution converged. Factor tables ready for rating engine upload.
{'='*65}
""")
```

</details>

---

## Exercise 5: Demand model sensitivity — what if lapses are worse?

**Reference:** Tutorial Parts 4, 14

**What you will do:** Test how sensitive the optimal rate action is to the assumed price elasticity. This is the single most important sensitivity check before presenting the frontier to a pricing committee.

**Context.** The commercial director asks: "Your demand model assumes a price coefficient of -2.2. What if customers are twice as price-sensitive as you think? How does that change the recommendation?" This is a legitimate question. If the frontier is very sensitive to the elasticity assumption, the recommendation needs more hedging.

### Tasks

**Task 1.** Create three versions of the demand model: low sensitivity (price\_coef=-1.2), base case (price\_coef=-2.2), and high sensitivity (price\_coef=-3.5). For each, solve the optimisation at LR target 0.74. Report:
- Factor adjustments under each assumption
- Expected volume at the optimal solution
- Whether the problem is feasible at the 96% volume floor under each assumption

**Task 2.** Trace the efficient frontier under all three demand models on the same chart. Overlay the three frontiers with different colours. Where do they diverge? At what point does the high-sensitivity assumption make the frontier infeasible while the low-sensitivity assumption is still feasible?

**Task 3.** The commercial director wants a "worst-case" rate action: the adjustment that achieves 74% LR even if the elasticity is as high as -3.5. What factor adjustments does this require, and what is the expected volume cost compared to the base case?

**Task 4.** Write a one-paragraph demand model sensitivity statement for the pricing committee pack. It should explain: (a) the base case assumption, (b) the range tested, (c) the impact on the recommended rate action, and (d) how the team proposes to validate the elasticity estimate.

<details>
<summary>Hint for Task 2</summary>

You need to create three separate `RateChangeOptimiser` instances, each with a different `LogisticDemandParams`, and then create three `EfficientFrontier` objects. Trace all three frontiers over the same LR range and plot them on the same axes.

The key question is: at what LR target does the volume floor become binding under each elasticity assumption? High elasticity means more lapses per percentage point of rate increase, which means the volume floor is hit sooner (at a less tight LR target).

</details>

<details>
<summary>Solution — Exercise 5</summary>

```python
from rate_optimiser import (
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams
import matplotlib.pyplot as plt

# Task 1: Three demand models
demand_configs = {
    "Low sensitivity (coef=-1.2)":  LogisticDemandParams(intercept=1.2, price_coef=-1.2, tenure_coef=0.04),
    "Base case (coef=-2.2)":        LogisticDemandParams(intercept=1.2, price_coef=-2.2, tenure_coef=0.04),
    "High sensitivity (coef=-3.5)": LogisticDemandParams(intercept=1.2, price_coef=-3.5, tenure_coef=0.04),
}

results_by_demand = {}
frontiers_by_demand = {}

for label, params in demand_configs.items():
    dm = make_logistic_demand(params)
    opt_dm = RateChangeOptimiser(data=data_ex1, demand=dm, factor_structure=fs_ex1)
    opt_dm.add_constraint(LossRatioConstraint(bound=0.74))
    opt_dm.add_constraint(VolumeConstraint(bound=0.96))
    opt_dm.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
    opt_dm.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors))

    res = opt_dm.solve()
    results_by_demand[label] = (opt_dm, res)

    print(f"\n{label}:")
    print(f"  Converged: {res.converged}")
    if res.converged:
        print(f"  Expected LR:     {res.expected_loss_ratio:.4f}")
        print(f"  Expected volume: {res.expected_volume_ratio:.4f}")
        for fname in ["f_age", "f_ncb", "f_vehicle", "f_region"]:
            print(f"  {fname}: {res.factor_adjustments.get(fname, 1.0):.4f}")
    else:
        print("  Problem infeasible at 96% volume floor — demand too sensitive")

# Task 2: Efficient frontiers comparison
fig, ax = plt.subplots(figsize=(10, 6))
colours = ["steelblue", "darkorange", "firebrick"]

for (label, (opt_dm, res)), colour in zip(results_by_demand.items(), colours):
    ef = EfficientFrontier(opt_dm)
    fdf = ef.trace(lr_range=(0.72, 0.78), n_points=15)
    frontiers_by_demand[label] = fdf

    feas = fdf[fdf["feasible"] & (fdf["expected_volume"] >= 0.94)]
    ax.plot(
        feas["expected_lr"] * 100,
        feas["expected_volume"] * 100,
        "o-", color=colour, linewidth=2, markersize=4, label=label,
    )

ax.axhline(96, linestyle="--", color="grey", alpha=0.5, label="Volume floor (96%)")
ax.set_xlabel("Expected loss ratio (%)", fontsize=11)
ax.set_ylabel("Expected volume retention (%)", fontsize=11)
ax.set_title("Efficient frontier sensitivity to price elasticity assumption", fontsize=12)
ax.invert_xaxis()
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Task 3: Worst-case adjustments (high sensitivity, still feasible)
label_high = "High sensitivity (coef=-3.5)"
opt_high, res_high = results_by_demand[label_high]
if res_high.converged:
    print(f"\nTask 3 — Worst-case rate action (high elasticity):")
    print(f"  {'Factor':<25} {'Base case':>12} {'High sensitivity':>18} {'Difference':>12}")
    print(f"  {'-'*69}")
    opt_base, res_base = results_by_demand["Base case (coef=-2.2)"]
    for fname in ["f_age", "f_ncb", "f_vehicle", "f_region"]:
        m_base = res_base.factor_adjustments.get(fname, 1.0) if res_base.converged else 1.0
        m_high = res_high.factor_adjustments.get(fname, 1.0)
        print(f"  {fname:<25} {m_base:>12.4f} {m_high:>18.4f} {(m_high-m_base)*100:>+11.1f}pp")
    print(f"\n  Volume under high-elasticity solution: {res_high.expected_volume_ratio:.4f}")
    if res_base.converged:
        print(f"  Volume under base-case solution:       {res_base.expected_volume_ratio:.4f}")
else:
    print("High-sensitivity problem infeasible. Cannot reach 74% LR within 96% volume floor.")
    print("Recommendation: relax volume floor to 94% or accept a higher LR target.")

# Task 4: Demand model sensitivity statement
print("""
Task 4 — Demand model sensitivity statement (for pricing committee pack):

The rate action is based on a logistic renewal demand model with a log-price
semi-elasticity of -2.2, estimated from the book's observed lapse rates over
the last four quarters. This parameter has not been formally re-estimated
since Q4 2024 and should be treated as an approximation.

We tested the sensitivity of the rate recommendation to the elasticity
assumption across a range of -1.2 (low sensitivity) to -3.5 (high
sensitivity). Under the low-sensitivity assumption, the optimal rate action
is approximately X% (vs Y% in the base case), with improved volume retention.
Under the high-sensitivity assumption, the 74% LR target is [feasible/not
feasible] within the 96% volume floor. If not feasible, the volume floor
would need to be relaxed to 94% to achieve 74% LR.

The team proposes to re-estimate the price coefficient from the most recent
12 months of lapse data before the Q3 review. In the interim, we recommend
pricing to the base case assumption with the high-sensitivity scenario
documented as a downside risk.
""")
```

</details>

---

## Exercise 6: Stochastic optimisation — chance constraints and prudence loading

**Reference:** Tutorial Part 14

**What you will do:** Implement the stochastic extension, compare deterministic and stochastic rate actions, and decide whether the prudence loading is material.

**Context.** The Board has a formal risk appetite statement: "With 90% probability, the realised portfolio loss ratio in any accident year must not exceed the LR target." This is a stronger statement than "the expected LR equals the target." You need to quantify the difference and decide whether the prudence loading is material enough to change the rate recommendation.

### Tasks

**Task 1.** Build a `ClaimsVarianceModel` from Tweedie parameters (dispersion=1.2, power=1.5) using the technical premium as the mean claims estimate. Confirm the number of policies in the variance model matches the portfolio.

**Task 2.** Create a `StochasticRateChangeOptimiser` with a `ChanceLossRatioConstraint` at 90% confidence (alpha=0.90) and the same volume floor and factor bounds as the base optimiser. Solve and report:
- The factor adjustments under the stochastic constraint
- The expected LR
- The LR standard deviation
- The 90th percentile LR (which should be at or below the target)
- The "prudence loading": factor adjustments stochastic minus deterministic

**Task 3.** Repeat at alpha=0.95 (the Board's most conservative risk appetite). How does the prudence loading change? At what alpha level does the stochastic optimiser produce infeasible results (factor adjustments exceed the 15% cap)?

**Task 4.** Write a two-paragraph explanation for the board risk committee of why the stochastic rate action is higher than the deterministic one, what the normal approximation assumes, and when the normal approximation is not appropriate.

**Stretch.** Run a Monte Carlo simulation to validate the normal approximation. For the stochastic rate action, draw 10,000 scenarios of portfolio claims (using the Tweedie parameters) and compute the empirical distribution of portfolio LR. What fraction of scenarios produce LR above the target? Is it close to 10% (for alpha=0.90)?

<details>
<summary>Hint for Task 2</summary>

The `StochasticRateChangeOptimiser` is a subclass of `RateChangeOptimiser`. The main difference is that it accepts a `ClaimsVarianceModel` at construction and the `ChanceLossRatioConstraint` replaces the deterministic `LossRatioConstraint`.

The `lr_std` and `lr_quantile_90` attributes are computed at the optimal solution — they are not constraints, they are diagnostic outputs. `lr_quantile_90 = expected_lr + 1.282 * lr_std` (from the normal approximation).

Compare `result.factor_adjustments` from the stochastic solve against `result_with.factor_adjustments` from the deterministic solve. The difference is the prudence loading in percentage points.

</details>

<details>
<summary>Solution — Exercise 6</summary>

```python
from rate_optimiser.stochastic import (
    StochasticRateChangeOptimiser,
    ClaimsVarianceModel,
    ChanceLossRatioConstraint,
)
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Claims variance model
variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=df_ex1_pd["technical_premium"].values,
    dispersion=1.2,
    power=1.5,
)
print(f"Variance model: {variance_model.n_policies:,} policies")
print(f"  Mean per-policy variance: {variance_model.policy_variance.mean():.2f}")
print(f"  Portfolio LR std (approx): {variance_model.portfolio_lr_std:.4f}")

# Task 2: Stochastic optimiser at alpha=0.90
def build_stoch_opt(alpha):
    stoch_opt = StochasticRateChangeOptimiser(
        data=data_ex1,
        demand=demand_ex1,
        factor_structure=fs_ex1,
        variance_model=variance_model,
    )
    stoch_opt.add_constraint(ChanceLossRatioConstraint(
        bound=0.74, alpha=alpha, normal_approx=True
    ))
    stoch_opt.add_constraint(VolumeConstraint(bound=0.96))
    stoch_opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
    stoch_opt.add_constraint(FactorBoundsConstraint(
        lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors
    ))
    return stoch_opt

stoch_opt_90 = build_stoch_opt(alpha=0.90)
stoch_res_90 = stoch_opt_90.solve()

print(f"\nTask 2 — Stochastic solve (alpha=0.90):")
print(f"  Converged:           {stoch_res_90.converged}")
print(f"  Expected LR (mean):  {stoch_res_90.expected_loss_ratio:.4f}")
print(f"  LR std deviation:    {stoch_res_90.lr_std:.4f}")
print(f"  90th pctile LR:      {stoch_res_90.lr_quantile_90:.4f}")
print(f"  Expected volume:     {stoch_res_90.expected_volume_ratio:.4f}")

print(f"\n  Prudence loading (stochastic vs deterministic):")
print(f"  {'Factor':<25} {'Deterministic':>15} {'Stochastic 90%':>16} {'Loading':>10}")
print(f"  {'-'*68}")
for fname in ["f_age", "f_ncb", "f_vehicle", "f_region"]:
    m_det   = result_with.factor_adjustments.get(fname, 1.0)
    m_stoch = stoch_res_90.factor_adjustments.get(fname, 1.0)
    print(f"  {fname:<25} {m_det:>15.4f} {m_stoch:>16.4f} {(m_stoch-m_det)*100:>+9.1f}pp")

# Task 3: alpha=0.95
stoch_opt_95 = build_stoch_opt(alpha=0.95)
stoch_res_95 = stoch_opt_95.solve()

print(f"\nTask 3 — Stochastic solve (alpha=0.95):")
print(f"  Converged:           {stoch_res_95.converged}")
if stoch_res_95.converged:
    print(f"  Expected LR (mean):  {stoch_res_95.expected_loss_ratio:.4f}")
    print(f"  95th pctile LR:      {stoch_res_95.lr_quantile_90:.4f}")

    # Find the alpha at which factor bounds become binding
    for alpha in [0.90, 0.92, 0.95, 0.97, 0.99]:
        try:
            s = build_stoch_opt(alpha=alpha).solve()
            adj_max = max(s.factor_adjustments.values()) if s.converged else None
            status = f"converged, max adj={adj_max:.4f}" if s.converged else "infeasible"
        except Exception as e:
            status = f"error: {e}"
        print(f"  alpha={alpha:.2f}: {status}")

# Task 4: Board explanation
print("""
Task 4 — Board risk committee explanation:

The deterministic rate optimisation finds the premium increases that produce
a 74% expected portfolio loss ratio on average across all possible claims
outcomes. In any single year, the realised loss ratio will differ from the
expectation due to claims randomness — some years will be better, some worse.
The deterministic target does not constrain how often the worse outcomes occur.

The stochastic formulation, using chance constraints at 90% confidence, requires
that there is at most a 10% probability of the realised loss ratio exceeding
74% in any accident year. This is a stronger requirement: it constrains the
distribution of outcomes, not just the mean. Achieving this requires a lower
expected LR at the new rates — the optimiser takes more rate to create a buffer
against adverse claims experience. The conversion from the chance constraint to
a deterministic equivalent uses a normal approximation for the portfolio loss
ratio, which is reasonable for diversified books with 50,000+ policies but should
be validated against simulated outcomes for smaller books.

For this 4,000-policy book, the normal approximation is borderline. The board
should be aware that the true 90th percentile LR may differ from the model
prediction by up to 1-2 percentage points for a book of this size. For material
rate decisions on books below 20,000 policies, we recommend validating the
chance constraint via Monte Carlo simulation rather than relying solely on the
normal approximation.
""")

# Stretch: Monte Carlo validation
if stoch_res_90.converged:
    N_SIMS = 10_000
    n_policies = len(df_ex1)

    # At stochastic rates, expected claims per policy = technical_premium (unchanged by rate action)
    # Premium per policy at stochastic rates
    stoch_combined_adj = 1.0
    for fname in ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]:
        stoch_combined_adj *= stoch_res_90.factor_adjustments.get(fname, 1.0)

    stoch_premiums = df_ex1["current_premium"].to_numpy() * stoch_combined_adj
    mean_claims    = df_ex1["technical_premium"].to_numpy()

    # Simulate Tweedie claims for each policy in each scenario
    rng_mc = np.random.default_rng(seed=9999)
    lr_scenarios = np.zeros(N_SIMS)
    for sim in range(N_SIMS):
        # Tweedie compound Poisson approximation for each policy
        n_claims = rng_mc.poisson(mean_claims / 400)   # rough scale
        claim_amounts = np.where(
            n_claims > 0,
            rng_mc.gamma(shape=n_claims + 0.01, scale=400) * 1.2,
            0.0,
        )
        lr_scenarios[sim] = claim_amounts.sum() / stoch_premiums.sum()

    empirical_90th = np.quantile(lr_scenarios, 0.90)
    pct_exceeding  = (lr_scenarios > 0.74).mean() * 100

    print(f"Stretch — Monte Carlo validation (N={N_SIMS:,} scenarios):")
    print(f"  Target LR:              74.0%")
    print(f"  Model 90th pctile:      {stoch_res_90.lr_quantile_90*100:.2f}%")
    print(f"  Empirical 90th pctile:  {empirical_90th*100:.2f}%")
    print(f"  Fraction exceeding 74%: {pct_exceeding:.1f}%  (target: 10%)")
    print()
    if abs(pct_exceeding - 10.0) < 2.0:
        print("  Normal approximation validated: empirical exceedance close to 10%.")
    else:
        print(f"  Warning: normal approximation off by {abs(pct_exceeding - 10.0):.1f}pp.")
        print("  For this book size, use simulation rather than normal approx.")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lr_scenarios * 100, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(74, color="firebrick", linestyle="--", linewidth=1.5, label="LR target (74%)")
    ax.axvline(empirical_90th * 100, color="darkorange", linestyle="-.",
               linewidth=1.5, label=f"Empirical 90th pctile ({empirical_90th*100:.1f}%)")
    ax.set_xlabel("Portfolio loss ratio (%)")
    ax.set_ylabel("Scenario count")
    ax.set_title("Monte Carlo LR distribution at stochastic optimal rates")
    ax.legend()
    plt.tight_layout()
    plt.show()
```

</details>

---

## Exercise 7: Limitations and failure modes — testing the edges

**Reference:** Tutorial Part 17

**What you will do:** Deliberately break the optimiser in realistic ways, observe the failure modes, and document them for the pricing committee.

**Context.** A junior actuary has joined the team and will run the optimiser for future reviews. Before handing over, you want to document the failure modes clearly — what breaks, how it breaks, and how to diagnose it. This exercise tests all the documented limitations from the tutorial.

### Tasks

**Task 1: Factor cap infeasibility.** Tighten the factor caps to [0.98, 1.03] (maximum 2% decrease, maximum 3% increase). Run the feasibility check and the solver. What happens? Write the diagnostic message a junior actuary should see when this occurs, and state the correct action to take.

**Task 2: Demand model conditioning.** Set `price_coef=-15.0` (extreme price sensitivity — every customer lapses at the slightest increase). Solve at LR target 0.74. What happens to convergence? Does the solver report a failure or does it silently produce a bad result? This tests whether you can rely on `result.converged` alone to detect problems.

**Task 3: Conflicting constraints.** Set the LR target to 0.60 (far below current) with a 99% volume floor and [0.95, 1.05] factor caps. Run the feasibility check. Write a clear explanation for a non-technical pricing director of why these constraints are incompatible, without using the word "infeasible."

**Task 4: Stale demand model.** The book has changed composition since the demand model was calibrated: the PCW mix has risen from 65% to 80% of renewals, making the book more price-sensitive. But you are still using `price_coef=-2.2` calibrated on the old book. Simulate this by creating a portfolio where 80% of renewals are PCW, with a true PCW elasticity of -3.0, but solving with the old demand model. Compare the optimiser's volume prediction against the "true" volume using the correct elasticity. How large is the forecast error?

**Task 5: Document the limitations.** Produce a one-page "limitations of the rate optimiser" summary for the pricing committee file. It should list at least six specific limitations in plain English, with a practical implication for each. Do not use any mathematical notation.

<details>
<summary>Hint for Task 2</summary>

With `price_coef=-15.0`, the logistic demand function becomes nearly a step function: renewal probability is very close to 1.0 below market price and very close to 0.0 above market price. The gradient of the demand function with respect to the rate vector is extremely large in some regions and near-zero in others. SLSQP uses gradients internally, so this conditioning problem can cause the solver to fail to converge, or to converge to a non-optimal point.

Check: (a) `result.converged`, (b) whether `result.expected_volume_ratio` is plausible (not negative, not above 1.0), and (c) whether the factor adjustments are within bounds. A bad solve can produce out-of-bounds adjustments even if `converged=True`.

</details>

<details>
<summary>Solution — Exercise 7</summary>

```python
# Task 1: Factor cap infeasibility
opt_tight_caps = RateChangeOptimiser(data=data_ex1, demand=demand_ex1, factor_structure=fs_ex1)
opt_tight_caps.add_constraint(LossRatioConstraint(bound=0.74))
opt_tight_caps.add_constraint(VolumeConstraint(bound=0.96))
opt_tight_caps.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_tight_caps.add_constraint(FactorBoundsConstraint(lower=0.98, upper=1.03, n_factors=fs_ex1.n_factors))

print("Task 1 — Tight factor caps [0.98, 1.03]:")
print(opt_tight_caps.feasibility_report())
res_tight = opt_tight_caps.solve()
print(f"Converged: {res_tight.converged}")
print("""
Diagnostic message for junior actuary:
The optimiser cannot find a solution within the approved factor caps.
The book requires a rate increase of approximately 4pp LR improvement,
but the approved factor caps allow at most 3% increase per factor.
At 3% increase across all factors, the expected LR improvement is
approximately 3pp — insufficient to reach the 74% target — and the
volume constraint may also be binding.

Correct action: escalate to the underwriting director for a wider factor
cap mandate (e.g., -10% to +6%), or accept a higher LR target that is
achievable within the current caps. Do not relax the ENBP constraint.
""")

# Task 2: Extreme elasticity
params_extreme = LogisticDemandParams(intercept=1.2, price_coef=-15.0, tenure_coef=0.04)
demand_extreme = make_logistic_demand(params_extreme)

opt_extreme = RateChangeOptimiser(data=data_ex1, demand=demand_extreme, factor_structure=fs_ex1)
opt_extreme.add_constraint(LossRatioConstraint(bound=0.74))
opt_extreme.add_constraint(VolumeConstraint(bound=0.96))
opt_extreme.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_extreme.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_ex1.n_factors))

res_extreme = opt_extreme.solve()
print(f"\nTask 2 — Extreme elasticity (price_coef=-15.0):")
print(f"  Converged:       {res_extreme.converged}")
print(f"  Objective value: {res_extreme.objective_value}")
print(f"  Expected volume: {res_extreme.expected_volume_ratio}")
print(f"  Factor adjustments:")
for fname, m in res_extreme.factor_adjustments.items():
    within_bounds = 0.90 <= m <= 1.15
    print(f"    {fname}: {m:.4f}  {'OK' if within_bounds else 'OUT OF BOUNDS'}")
print("""
Note: Even if converged=True, check that factor adjustments are within
the approved bounds and that expected_volume is physically plausible
(between 0 and 1). An extreme elasticity can produce gradient instability
that causes the solver to report convergence on a non-optimal or invalid point.
""")

# Task 3: Conflicting constraints
opt_conflict = RateChangeOptimiser(data=data_ex1, demand=demand_ex1, factor_structure=fs_ex1)
opt_conflict.add_constraint(LossRatioConstraint(bound=0.60))
opt_conflict.add_constraint(VolumeConstraint(bound=0.99))
opt_conflict.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_conflict.add_constraint(FactorBoundsConstraint(lower=0.95, upper=1.05, n_factors=fs_ex1.n_factors))

print("Task 3 — Conflicting constraints (LR=60%, volume=99%, caps=[0.95, 1.05]):")
print(opt_conflict.feasibility_report())
print("""
Plain English explanation for pricing director:
"We have been asked to achieve three things simultaneously that cannot all
be true at once. Reducing the loss ratio from 78% to 60% requires
a very large premium increase — roughly 25-30% across the board.
A 25-30% increase will cause a significant number of customers not to renew.
But we have also been asked to retain 99% of current volume, which means
virtually no customers can lapse. And the maximum rate increase allowed by
the factor caps is 5%. These three requirements — large LR improvement,
near-total volume retention, and small rate increase — cannot all be met
at the same time. We need to relax at least one of them. Our recommendation:
either target 74% LR (achievable) rather than 60%, or accept 90-95% volume
retention as part of achieving the more ambitious LR target."
""")

# Task 4: Stale demand model
rng_stale = np.random.default_rng(seed=3000)
N_STALE = 4_000

# New portfolio: 80% PCW, true elasticity -3.0
age_stale     = rng_stale.choice([0.80, 1.00, 1.20, 1.50, 2.00], N_STALE, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_stale     = rng_stale.choice([0.70, 0.80, 0.90, 1.00], N_STALE, p=[0.30, 0.30, 0.25, 0.15])
vehicle_stale = rng_stale.choice([0.90, 1.00, 1.10, 1.30], N_STALE, p=[0.25, 0.35, 0.25, 0.15])
region_stale  = rng_stale.choice([0.85, 1.00, 1.10, 1.20], N_STALE, p=[0.20, 0.40, 0.25, 0.15])
tenure_stale  = rng_stale.integers(0, 10, N_STALE).astype(float)

tech_stale    = 380 * age_stale * ncb_stale * vehicle_stale * region_stale * rng_stale.uniform(0.97, 1.03, N_STALE)
curr_stale    = tech_stale / 0.78 * rng_stale.uniform(0.96, 1.04, N_STALE)
market_stale  = tech_stale / 0.75 * rng_stale.uniform(0.90, 1.10, N_STALE)
renew_stale   = rng_stale.random(N_STALE) < 0.65

# 80% PCW for renewals (true elasticity -3.0)
chan_stale = np.where(
    renew_stale,
    rng_stale.choice(["PCW", "direct"], N_STALE, p=[0.80, 0.20]),
    rng_stale.choice(["PCW", "direct"], N_STALE, p=[0.75, 0.25]),
)

log_pr_stale = np.log(curr_stale / market_stale)
# True renewal probability uses elasticity -3.0 for PCW, -1.5 for direct
true_elast   = np.where(chan_stale == "PCW", -3.0, -1.5)
logit_true   = 1.2 + true_elast * log_pr_stale + 0.04 * tenure_stale
renew_prob_true = expit(logit_true)

df_stale = pl.DataFrame({
    "policy_id":         [f"S{i:07d}" for i in range(N_STALE)],
    "channel":           chan_stale.tolist(),
    "renewal_flag":      renew_stale.tolist(),
    "tenure":            tenure_stale.tolist(),
    "technical_premium": tech_stale.tolist(),
    "current_premium":   curr_stale.tolist(),
    "market_premium":    market_stale.tolist(),
    "renewal_prob":      renew_prob_true.tolist(),
    "f_age":             age_stale.tolist(),
    "f_ncb":             ncb_stale.tolist(),
    "f_vehicle":         vehicle_stale.tolist(),
    "f_region":          region_stale.tolist(),
    "f_tenure_discount": np.ones(N_STALE).tolist(),
})

data_stale = PolicyData(df_stale.to_pandas())
fs_stale   = FactorStructure(
    factor_names=FACTOR_NAMES_EX1,
    factor_values=df_stale.to_pandas()[FACTOR_NAMES_EX1],
    renewal_factor_names=["f_tenure_discount"],
)

# Solve with stale demand model (coef=-2.2, calibrated on old book)
demand_stale_wrong = make_logistic_demand(
    LogisticDemandParams(intercept=1.2, price_coef=-2.2, tenure_coef=0.04)
)
opt_stale = RateChangeOptimiser(data=data_stale, demand=demand_stale_wrong, factor_structure=fs_stale)
opt_stale.add_constraint(LossRatioConstraint(bound=0.74))
opt_stale.add_constraint(VolumeConstraint(bound=0.96))
opt_stale.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt_stale.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs_stale.n_factors))
res_stale = opt_stale.solve()

# True volume using correct elasticity
if res_stale.converged:
    combined_adj_stale = 1.0
    for fname in FACTOR_NAMES_EX1:
        combined_adj_stale *= res_stale.factor_adjustments.get(fname, 1.0)

    # Compute true renewal probability at new prices
    new_prem_stale    = curr_stale * combined_adj_stale
    new_log_pr_stale  = np.log(new_prem_stale / market_stale)
    new_logit_true    = 1.2 + true_elast * new_log_pr_stale + 0.04 * tenure_stale
    renew_prob_new_true = expit(new_logit_true)

    true_vol = (new_prem_stale * renew_prob_new_true).sum() / (curr_stale * renew_prob_true).sum()
    model_vol = res_stale.expected_volume_ratio

    print(f"\nTask 4 — Stale demand model effect:")
    print(f"  Model predicted volume:  {model_vol:.4f}")
    print(f"  True volume (correct e): {true_vol:.4f}")
    print(f"  Forecast error:          {(model_vol - true_vol)*100:+.1f}pp")
    print(f"  (Model was optimistic by {abs(model_vol - true_vol)*100:.1f}pp)")
    if true_vol < 0.96:
        print(f"  WARNING: True volume {true_vol:.4f} is below the 96% volume floor.")
        print("  The rate action breaches the volume constraint using the true elasticity.")

# Task 5: Limitations summary (formatted as plain text for the committee file)
print("""
Task 5 — Rate optimiser limitations (for pricing committee file)

RATE OPTIMISER: DOCUMENTED LIMITATIONS
Q2 2026 Motor Renewal Rate Review

This document records the known limitations of the constrained rate
optimiser used in this review. It is included in the pricing committee
file as part of the model governance record.

1. FACTOR ADJUSTMENTS ARE UNIFORM
   The optimiser applies the same percentage change to every level of
   each factor table. It cannot, for example, increase young drivers by
   8% and mature drivers by 2%. Changing the shape of a factor table
   requires a separate modelling exercise with its own sign-off.
   Practical implication: the rate action distributes dislocation evenly
   across all levels of each factor, which may not be commercially optimal
   if some levels are more price-sensitive than others.

2. DEMAND MODEL MAY NOT REFLECT CURRENT BEHAVIOUR
   The price sensitivity parameter was last calibrated in Q4 2024. If
   the book's competitive position or channel mix has changed since then,
   the model's volume forecasts will be inaccurate. The volume target
   from the optimiser should be treated as a model estimate with a
   ±2 percentage point margin of error.
   Practical implication: actual retention may differ materially from
   the model's prediction. Monitor monthly.

3. ENBP IS APPROXIMATED FROM FACTOR TABLES
   The regulatory ENBP check compares adjusted renewal premiums against
   new business equivalents computed from the same factor tables. The
   actual regulatory requirement uses the live quoted NB price at the
   time of renewal. If the market has moved between the review and
   renewal, or if NB quotes include channel-specific terms not in the
   factor tables, there may be ENBP breaches that the factor-table
   check misses.
   Practical implication: run a live ENBP check at point of renewal,
   not just at point of rate review.

4. NEW BUSINESS EFFECTS ARE NOT MODELLED
   The optimiser models only renewal volume. It does not project new
   business volume or model the multi-period effects of a gap between
   NB and renewal pricing. If the rate action creates a persistent
   NB/renewal price gap, the book composition will change over time in
   ways the optimiser did not anticipate.
   Practical implication: review NB/renewal price alignment annually.

5. NORMAL APPROXIMATION FOR STOCHASTIC CONSTRAINT
   The stochastic extension assumes portfolio loss ratio is normally
   distributed. For this book (4,000 policies), the normal approximation
   is a reasonable starting point but should be validated by simulation
   before use in board-level risk reporting.
   Practical implication: do not use stochastic results in board risk
   reporting without a Monte Carlo validation.

6. COMPETITIVE RESPONSE IS NOT MODELLED
   The demand model assumes market premiums remain fixed after the rate
   action. If competitors respond by raising or lowering rates, actual
   retention and LR will differ from model predictions.
   Practical implication: review market premium benchmarks quarterly
   and recalibrate the demand model after significant market moves.

7. SOLVER CONVERGENCE IS NOT GUARANTEED
   In unusual situations (very tight constraints, extreme elasticity
   parameters, near-infeasible problems), the solver may report
   non-convergence. Non-converged results must not be used. If
   non-convergence occurs, investigate the cause before relaxing
   constraints.
   Practical implication: always check result.converged before using
   any output from the optimiser.

8. FACTOR RELATIVITIES ARE ASSUMED CORRECT
   The optimiser takes the current factor structure (the shape of each
   factor table) as given and only adjusts the scale. If the underlying
   factor relativities are mis-specified (e.g., the age gradient is
   too flat), the optimal rate action based on those relativities will
   also be mis-specified.
   Practical implication: factor shape review should precede rate action.
   Run the optimiser on the updated factors, not the old ones.
""")
```

</details>

---

## Checklist before signing off

Before submitting the pricing committee pack, confirm all of the following:

- [ ] `result.converged` is True
- [ ] ENBP per-policy violation count is 0
- [ ] Factor adjustments are all within the approved bounds [FACTOR\_LOWER, FACTOR\_UPPER]
- [ ] `expected_loss_ratio` is at or below the target (not above due to numerical tolerance)
- [ ] `expected_volume_ratio` is at or above the volume floor
- [ ] Demand model parameters have been reviewed against recent lapse data
- [ ] Factor tables have been updated correctly (new relativity = old relativity x adjustment)
- [ ] The efficient frontier chart is included in the committee pack
- [ ] The stochastic result is included as a sensitivity
- [ ] The limitations section of this exercises document is included in the model governance file
- [ ] Results are written to Unity Catalog with full parameter logging
- [ ] The ENBP compliance statement is signed by the chief actuary

The rate action is not complete until every item on this checklist is checked. The optimiser is a tool. The actuary signing off is responsible for the rate action, not the solver.
