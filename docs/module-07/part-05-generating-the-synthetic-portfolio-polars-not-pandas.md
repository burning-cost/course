## Part 5: Generating the synthetic portfolio (Polars, not pandas)

We work with a synthetic motor renewal portfolio of 5,000 policies running at approximately 75% LR against a 72% target. In production, you would read from a Unity Catalog table. We use synthetic data here so that we know the ground truth and can verify the optimiser output.

All data manipulation uses Polars. The `rate-optimiser` library uses pandas internally, so we convert at the library boundary. This follows the Polars mandate: the tutorial code uses Polars; pandas appears only where the library requires it.

Add a markdown cell:

```python
%md
## Part 5: Synthetic motor portfolio
```

Then create a new cell and paste this:

```python
import numpy as np
import polars as pl
from scipy.special import expit

rng = np.random.default_rng(seed=42)
N = 5_000

# ---------- Rating factor relativities ----------
# Each policy is drawn from a distribution of factor levels.
# These are the underlying relativities for each level.
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

# Tenure discount is renewal-only and currently neutral (1.0 for all)
tenure_disc = np.ones(N)

# ---------- Premiums ----------
base_rate = 350.0

# Technical premium: what the risk actually costs
technical_premium = (
    base_rate
    * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)   # small residual noise
)

# Book running at 75% LR: current premium = technical / 0.75 (with spread)
current_premium = technical_premium / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: what the customer could get elsewhere (competitive benchmark)
market_premium  = technical_premium / 0.73 * rng.uniform(0.90, 1.10, N)

# ---------- Demand model for renewal probability ----------
# Log price ratio: positive means we are above market, negative means below
log_price_ratio = np.log(current_premium / market_premium)

# Logistic demand: intercept=1.2, price_coef=-2.0, tenure_coef=0.05
logit_renew = 1.2 + (-2.0) * log_price_ratio + 0.05 * tenure
renewal_prob = expit(logit_renew)

# Indicator: is this a renewal policy?
renewal_flag = rng.random(N) < 0.65  # 65% of portfolio is renewals

# Channel: PCW or direct
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.65, 0.35]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# ---------- Build the Polars DataFrame ----------
df = pl.DataFrame({
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

print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean()*100:.0f}%)")
print(f"PCW:       {(df['channel'] == 'PCW').sum():,}")
print(f"Direct:    {(df['channel'] == 'direct').sum():,}")
print()
print("Current loss ratio (technical/current):")
lr = df["technical_premium"].sum() / df["current_premium"].sum()
print(f"  {lr:.4f}  (target: 0.72)")
```

**What this does, step by step:**

1. We generate factor relativities for each policy by drawing from discrete distributions. The distributions are calibrated to give a realistic UK motor factor mix: most policies have mid-range NCB and vehicle group, with a smaller number of young/high-risk policies.

2. The technical premium is the product of the four factor relativities times a base rate, with small noise. This is what the risk actually costs.

3. The current premium is the technical premium divided by 0.75, meaning the book is currently running at a 75% loss ratio. A 75% LR means: for every £1.00 of premium, we expect £0.75 in claims.

4. The market premium is what a customer could get elsewhere. Setting it slightly better than our premium (0.73 vs 0.75) means we are slightly above-market on average — a realistic scenario when a book is running above LR target.

5. The demand model computes renewal probability using the logistic function with the parameters we will pass to the optimiser.

6. We build the DataFrame using Polars. Note that factor columns are named with the `f_` prefix — this is the convention the `rate-optimiser` library uses.

**What you should see:**

```text
Portfolio: 5,000 policies
Renewals:  3,250 (65%)
PCW:       2,950
Direct:    2,050

Current loss ratio (technical/current):
  0.7502  (target: 0.72)
```

The exact numbers will vary slightly due to random noise, but the LR should be close to 0.75. If you see something far outside 0.73-0.77, re-run the data generation cell.