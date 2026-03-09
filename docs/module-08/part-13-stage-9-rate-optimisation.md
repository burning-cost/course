## Part 13: Stage 9 -- Rate optimisation

The rate optimiser from Module 7 takes a renewal portfolio and finds the factor adjustments that minimise loss ratio subject to volume and ENBP constraints. In a well-connected pipeline, the inputs to the rate optimiser come from the upstream stages: the frequency model's pure premium predictions define the technical premium, and the SHAP relativities define the factor structure.

Add a markdown cell:

```python
%md
## Stage 9: Rate optimisation -- connected to Stage 6 and Stage 7
```

### Building the renewal portfolio from model predictions

The renewal portfolio is the set of policies coming up for renewal in the next rating period. In production, this comes from the policy administration system. Here, we use the test set predictions as a proxy -- the test year represents the most recent pricing period, and its policies are the renewal candidates.

```python
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint, ENBPConstraint,
    FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams
from scipy.special import expit
import pandas as pd

# -----------------------------------------------------------------------
# Build renewal portfolio.
#
# The connection to Stage 6:
#   - pure_premium comes from freq_model.predict * sev_model.predict
#     (not from a separate synthetic generation step)
#   - This ensures the rate optimiser is working from the same technical
#     premium numbers that informed the pricing review
#
# The connection to Stage 7:
#   - Factor structure uses the SHAP relativities directly
#   - Region SHAP relativities from Stage 7 define f_region
#   - NCB deficit SHAP relativities define f_ncb
#   - These are not synthetic proxies -- they are the model's own output
# -----------------------------------------------------------------------

n_renewal = min(5_000, len(df_test))   # cap for computation speed

rng_opt = np.random.default_rng(seed=9999)

# Select the renewal subset
ren_idx  = rng_opt.choice(len(df_test), n_renewal, replace=False)
df_ren   = df_test.iloc[ren_idx].copy()
pp_ren   = pure_premium[ren_idx]

# Technical premium: pure premium (frequency * severity)
tech_prem = pp_ren

# Current premium: assume book running at 75% LR
# In production, current_premium comes from the policy administration system
curr_prem = tech_prem / LR_TARGET

# Market premium: assume market is 2pp below current LR
mkt_prem  = tech_prem / (LR_TARGET - 0.02)

# Renewal probability: logistic model on price relativity
# P(renew) = sigmoid(intercept + price_coef * log(curr/mkt))
renewal_prob = expit(1.0 + (-2.0) * np.log(curr_prem / mkt_prem))

# -----------------------------------------------------------------------
# Factor structure -- derived from SHAP relativities
#
# For region: use the SHAP relativities computed in Stage 7.
# For NCB: map ncb_deficit to its SHAP relativity quintile.
#
# In a production pipeline, this mapping is automatic from the SHAP output.
# Here we implement it manually for clarity.
# -----------------------------------------------------------------------

# Region factor: from SHAP relativities (Stage 7)
region_shap = shap_relativities.get("region", {})
if region_shap:
    # Map each policy's region to its SHAP-derived relativity
    f_region = np.array([
        region_shap.get(str(r), 1.0)
        for r in df_ren["region"].values
    ])
else:
    # Fallback if SHAP relativities not available
    region_rel = {"North": 0.82, "Midlands": 0.90, "SouthEast": 1.08,
                  "London": 1.38, "SouthWest": 0.86}
    f_region = np.array([region_rel.get(str(r), 1.0) for r in df_ren["region"].values])

# NCB factor: FEATURE_COLS does not include ncb_deficit in this pipeline,
# so we default to a flat factor of 1.0. In a production model that includes
# ncb_deficit, you would map SHAP relativities from Stage 7 here.
f_ncb = np.ones(n_renewal)

# Age factor: from age_mid (continuous -- use quintile)
age_vals = df_ren["age_mid"].values if "age_mid" in df_ren.columns else np.full(n_renewal, 43.0)
age_relative = (age_vals - age_vals.mean()) / (age_vals.std() + 1e-6)
f_age = np.exp(0.20 * age_relative)   # approximate monotone age effect

print(f"Renewal portfolio: {n_renewal:,} policies")
print(f"Mean technical premium: £{tech_prem.mean():,.2f}")
print(f"Mean current premium:   £{curr_prem.mean():,.2f}")
print(f"Mean renewal probability: {renewal_prob.mean():.3f}")
```

### Running the optimiser

```python
channels = rng_opt.choice(["PCW", "direct"], n_renewal, p=[0.68, 0.32])

renewal_port = pd.DataFrame({
    "policy_id":         df_ren["policy_id"].values,
    "channel":           channels,
    "renewal_flag":      np.ones(n_renewal, dtype=bool),
    "technical_premium": tech_prem,
    "current_premium":   curr_prem,
    "market_premium":    mkt_prem,
    "renewal_prob":      renewal_prob,
    "tenure":            rng_opt.integers(0, 10, n_renewal).astype(float),
    "f_region":          f_region,
    "f_ncb":             f_ncb,
    "f_age":             f_age,
    "f_iat":             np.ones(n_renewal),   # introduced factor (no change)
})

data = PolicyData(renewal_port)
fs   = FactorStructure(
    factor_names=["f_region", "f_ncb", "f_age", "f_iat"],
    factor_values=renewal_port[["f_region", "f_ncb", "f_age", "f_iat"]],
    renewal_factor_names=["f_iat"],   # only f_iat is renewal-only
)

demand = make_logistic_demand(
    LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.04)
)
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

# Constraints from Module 7
opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15,
                                          n_factors=fs.n_factors))

result = opt.solve()
print(result.summary())
```

### Writing rate action factors to Delta

```python
rate_factors_df = pl.DataFrame({
    "run_date":      [RUN_DATE] * 4,
    "factor":        ["f_region", "f_ncb", "f_age", "f_iat"],
    "adjustment":    [float(result.factor_adjustments.get(f, 1.0)) for f in ["f_region", "f_ncb", "f_age", "f_iat"]],
    "lr_target":     [LR_TARGET] * 4,
    "optimiser_converged": [result.converged] * 4,
    "expected_lr":   [round(result.expected_loss_ratio, 4)] * 4,
    "expected_vol":  [round(result.expected_volume_ratio, 4)] * 4,
})

spark.createDataFrame(rate_factors_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["rate_change"])

print(f"Rate action factors written to {TABLES['rate_change']}")
print(f"Optimiser converged: {result.converged}")
print(f"Expected LR after action: {result.expected_loss_ratio:.3f}")
print(f"Expected volume retention: {result.expected_volume_ratio:.3f}")
```