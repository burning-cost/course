## Part 5: Generating the synthetic portfolio

We work with a synthetic motor renewal portfolio of 5,000 policies running at approximately 75% LR against a 72% target. In production, you would read from a Unity Catalog table. We use synthetic data here so that we know the ground truth and can verify the optimiser output.

All data manipulation uses Polars for analysis and output. The `insurance-optimise` library takes numpy arrays directly — no pandas conversion required.

Add a markdown cell:

```python
%md
## Part 5: Synthetic motor portfolio
```

Then create a new cell and paste this:

```python
import numpy as np
import polars as pl

rng = np.random.default_rng(seed=42)
N = 5_000

# ---------- Rating factor relativities ----------
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N,
                          p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N,
                          p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N,
                          p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N,
                          p=[0.20, 0.40, 0.25, 0.15])

# Tenure: 0-9 years with the insurer
tenure = rng.integers(0, 10, N).astype(float)

# ---------- Premiums ----------
base_rate = 350.0

# Expected loss cost: what the risk actually costs in claims
expected_loss_cost = (
    base_rate
    * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)   # small residual noise
)

# Technical price: cost plus expense loading.
# At 80% technical LR, technical_price = cost / 0.80.
technical_price = expected_loss_cost / 0.80

# Book running at 75% LR: current premium = cost / 0.75 (with spread)
current_premium = expected_loss_cost / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: what the customer could get elsewhere (competitive benchmark)
market_premium  = expected_loss_cost / 0.73 * rng.uniform(0.90, 1.10, N)

# ---------- Renewal flag and channel ----------
renewal_flag = rng.random(N) < 0.65   # 65% of portfolio is renewals
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.65, 0.35]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# ---------- Per-policy elasticity ----------
# PCW customers are more price-sensitive than direct customers.
elasticity = np.where(channel == "PCW", -2.0, -1.2)
elasticity = elasticity + 0.03 * tenure   # tenure stickiness
elasticity = np.clip(elasticity, -3.5, -0.5)

# ---------- Baseline demand probability ----------
# Renewal probability at current rates, from a logistic lapse model.
log_price_ratio = np.log(current_premium / market_premium)
logit_p = 1.2 + (-2.0) * log_price_ratio + 0.05 * tenure
p_demand = 1.0 / (1.0 + np.exp(-logit_p))
p_demand = np.clip(p_demand, 0.05, 0.95)

# ---------- ENBP (FCA PS21/11) ----------
# For renewal policies, the ENBP is the equivalent new business price.
# Set just above current premium to reflect a small NB discount.
enbp = np.where(renewal_flag, current_premium * rng.uniform(0.98, 1.05, N), 0.0)

# ---------- Build the Polars DataFrame ----------
df = pl.DataFrame({
    "policy_id":          [f"MTR{i:07d}" for i in range(N)],
    "channel":            channel.tolist(),
    "renewal_flag":       renewal_flag.tolist(),
    "tenure":             tenure.tolist(),
    "technical_price":    technical_price.tolist(),
    "expected_loss_cost": expected_loss_cost.tolist(),
    "current_premium":    current_premium.tolist(),
    "market_premium":     market_premium.tolist(),
    "p_demand":           p_demand.tolist(),
    "elasticity":         elasticity.tolist(),
    "enbp":               enbp.tolist(),
})

print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean()*100:.0f}%)")
print(f"PCW:       {(df['channel'] == 'PCW').sum():,}")
print(f"Direct:    {(df['channel'] == 'direct').sum():,}")
print()
print("Current loss ratio (expected_loss_cost / current_premium):")
lr = df["expected_loss_cost"].sum() / df["current_premium"].sum()
print(f"  {lr:.4f}  (target: 0.72)")
print()
print(f"Mean p_demand (renewal policies): {p_demand[renewal_flag.astype(bool)].mean():.3f}")
print(f"Mean elasticity:                  {elasticity.mean():.2f}")
```

**What this does, step by step:**

1. We generate rating factor relativities for each policy. These determine the underlying risk cost.

2. The expected loss cost is the product of four factor relativities times a base rate, with small noise. This is what the risk actually costs in claims.

3. The technical price is the expected loss cost divided by 0.80, reflecting an 80% technical LR (20% expense and profit loading). This is the reference price for multipliers.

4. The current premium is set at the cost / 0.75 level, meaning the book is running at approximately 75% LR — above the 72% target.

5. Per-policy elasticities vary by channel (PCW more elastic, direct less elastic) and tenure (longer-tenured customers slightly less elastic).

6. The baseline demand probability `p_demand` comes from a logistic lapse model applied at current premium levels. This is what the optimiser uses as `x0` in `x(m) = x0 * m^epsilon`.

7. ENBP is set just above the current premium for renewal policies, representing a small new business discount. Renewal premiums may not exceed this.

**What you should see:**

```text
Portfolio: 5,000 policies
Renewals:  3,250 (65%)
PCW:       2,950
Direct:    2,050

Current loss ratio (expected_loss_cost / current_premium):
  0.7500  (target: 0.72)

Mean p_demand (renewal policies): 0.761
Mean elasticity:                  -1.73
```

The exact numbers will vary slightly due to random noise, but the LR should be close to 0.75. If you see something far outside 0.73-0.77, re-run the data generation cell.
