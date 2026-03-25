# Module 9 Exercises: Demand Elasticity

Eight exercises. Work through them in order — each builds on data and fitted objects from earlier exercises. Solutions are at the end of each exercise in collapsed sections.

Before starting: read Parts 1–17. Every concept used here is explained there.

---

## Setup

Run this before Exercise 1. All exercises use these objects.

```python
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from insurance_causal.elasticity.data import make_renewal_data, true_gate_by_ncd
from insurance_causal.elasticity.fit import RenewalElasticityEstimator
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics
from insurance_causal.elasticity.surface import ElasticitySurface
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
from insurance_causal.elasticity.demand import demand_curve

import statsmodels.formula.api as smf

# Base renewal dataset
df = make_renewal_data(n=50_000, seed=42, price_variation_sd=0.08)
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

# Naive benchmark
df_pd = df.with_columns([
    pl.col("channel").cast(pl.Utf8),
    pl.col("vehicle_group").cast(pl.Utf8),
    pl.col("region").cast(pl.Utf8),
]).to_pandas()

formula = "renewed ~ log_price_change + age + ncd_years + C(vehicle_group) + C(region) + C(channel)"
naive_logit = smf.logit(formula, data=df_pd).fit(disp=0)
naive_coef = naive_logit.params["log_price_change"]

print(f"Setup complete. Naive coefficient: {naive_coef:.3f}")
print(f"True ATE: {df['true_elasticity'].mean():.3f}")
```

---

## Exercise 1: Diagnosing a weak treatment dataset

**Reference:** Part 6

**What you will do:** Generate a near-deterministic price dataset, run the treatment variation diagnostic, and confirm that the DML estimator gives unreliable results when treatment variation is insufficient.

### Setup

```python
df_ndp = make_renewal_data(n=50_000, seed=42, near_deterministic=True)
```

### Tasks

**Task 1.** Run `ElasticityDiagnostics().treatment_variation_report()` on `df_ndp` with the same confounders as the tutorial. Print `report.summary()`. Record the variation fraction and the nuisance R².

**Task 2.** Despite the warning, fit `RenewalElasticityEstimator(n_estimators=100, catboost_iterations=100)` on `df_ndp`. Compare the resulting ATE and 95% CI to the ATE and CI from the tutorial (good data, `price_variation_sd=0.08`). How do the CI widths compare?

**Task 3.** The `suggestions` attribute of the report lists remedies. Rank the five suggestions from easiest to implement in a real UK motor book (assuming you have 2 years of renewal history and a production renewal quoting system) to hardest.

<details>
<summary>Solution</summary>

**Task 1.** On `near_deterministic=True` data:
- `variation_fraction` ≈ 0.03–0.08 (below the 0.10 threshold)
- `nuisance_r2` ≈ 0.92–0.96 (above the 0.90 threshold)
- `weak_treatment=True`

**Task 2.** The ATE from the weak-treatment data will have a CI width 3–5× wider than the good data. It may include zero, making the estimate commercially useless. The point estimate is unreliable — it could be anything from −0.5 to −4.0 across different random seeds.

**Task 3.** Easiest to hardest:
1. Exploit bulk re-rating quasi-experiments (history already exists if you have had a bulk re-rate)
2. Use rate change timing heterogeneity (anniversary date variation exists across the book)
3. Use panel data with within-customer variation (requires 3+ renewal observations per customer — available if book is old enough)
4. Run randomised A/B price tests (requires technology change to quoting system; gold standard)
5. Exploit the PS21/5 kink (requires careful design and strong assumptions about selection at the ENBP boundary)
</details>

---

## Exercise 2: Fitting and validating the DML estimator

**Reference:** Parts 5–8

**What you will do:** Fit `RenewalElasticityEstimator` on the tutorial dataset and validate the ATE recovery against the known ground truth.

### Tasks

**Task 1.** Fit the estimator with `cate_model="linear_dml"` instead of `"causal_forest"`. Compare:
- The ATE and 95% CI
- The fit time
- The GATE by NCD band (use `.gate(df, by="ncd_years")`)

Does `LinearDML` recover the same GATE gradient as `CausalForestDML`?

**Task 2.** Using the `CausalForestDML` estimator from the tutorial (or refit), run `.gate(df, by="ncd_years")` and compute the absolute difference between the estimated GATE and the true GATE for each NCD band. Which NCD band has the largest estimation error? Why might that be?

**Task 3.** The `cate_interval()` method returns per-row 95% confidence intervals. Compute the mean CI width by NCD band. Are the CI widths correlated with segment size (the `n` column from `.gate()`)?

<details>
<summary>Solution</summary>

**Task 1.** `LinearDML` is substantially faster (seconds vs. minutes) but estimates a constant treatment effect — it cannot produce a GATE gradient. The `.gate()` call on a `LinearDML` estimator returns the same value for all NCD bands (the ATE). The ATE value should be similar to `CausalForestDML`.

**Task 2.** The largest estimation error is typically for NCD=0 or NCD=1 — young, high-risk customers who are less common in the book and whose price variation is more tightly constrained by the ENBP ceiling. Smaller sample sizes per segment and less treatment variation within the segment both increase estimation error.

**Task 3.** CI widths should be narrower for larger segments. NCD=3–4 (middle of the distribution) will tend to have the narrowest CIs. NCD=0 and NCD=5 may have wider CIs despite having reasonable counts, because the CATE distribution is more variable (extreme elasticities have more noise).
</details>

---

## Exercise 3: Building a two-dimensional elasticity surface

**Reference:** Part 10

**What you will do:** Use `ElasticitySurface` to build a segment summary across NCD years × channel, and identify the top three most elastic and top three least elastic segments.

### Tasks

**Task 1.** Using the `est` object from the tutorial (or a newly fitted one), create an `ElasticitySurface` and call `.segment_summary(df, by=["ncd_years", "channel"])`. Sort the result by `elasticity` (ascending — most elastic first). Which three segment combinations have the most negative elasticity? Which three have the least negative?

**Task 2.** Plot the NCD × channel surface using `.plot_surface(df, dims=["ncd_years", "channel"])`. Confirm that PCW channel cells are darker (more elastic) than direct channel cells at the same NCD band.

**Task 3.** The `elasticity_at_10pct` column gives the expected change in renewal probability for a 10% price increase. For the most elastic segment (from Task 1), what is the expected absolute reduction in renewal probability for a 10% increase? If that segment has 2,000 policies, how many additional lapses would a 10% increase generate?

<details>
<summary>Solution</summary>

**Task 1.** Most elastic: NCD=0 PCW, NCD=1 PCW, NCD=0 direct (in that order, roughly). Least elastic: NCD=5 broker, NCD=5 direct, NCD=4 direct.

**Task 2.** On the heatmap, all PCW columns should be darker than their corresponding direct or broker columns for the same NCD band. The 30% PCW elasticity amplification in the DGP should be visible as a consistent channel offset.

**Task 3.** For NCD=0 PCW with true elasticity ≈ −3.5 × 1.3 (PCW amplifier) ≈ −4.55, and log(1.1) = 0.0953, the expected renewal prob change is −4.55 × 0.0953 ≈ −0.43 pp. At a base renewal rate of around 70% for this high-elasticity segment, −0.43 pp represents 0.43% × 2,000 ≈ 9 additional lapses per year. The estimated CATE will be somewhat less extreme than the true value due to estimation noise.
</details>

---

## Exercise 4: Understanding ENBP constraint binding

**Reference:** Parts 12–13

**What you will do:** Analyse which segments have the most binding ENBP constraints and interpret the commercial implications.

### Tasks

**Task 1.** Using `RenewalPricingOptimiser` with `floor_loading=1.0`, run `optimise(df, objective="profit")`. Compute the fraction of ENBP-binding policies (headroom < £1) by NCD band and channel. Which segment combination has the highest proportion of binding ENBP constraints?

**Task 2.** For the segment with the highest binding proportion, what does this tell you about the profit opportunity? Is the elasticity model useful for this segment?

**Task 3.** Now run `optimise(df, objective="retention")`. Compare the mean `predicted_renewal_prob` under the two objectives. What price does the retention objective set, and why?

<details>
<summary>Solution</summary>

**Task 1.** The highest binding proportion will be in NCD=5, direct or broker channel — the least elastic customers. The optimiser wants to charge them more than the ENBP, but cannot. Binding fractions for this group can exceed 70% of policies.

**Task 2.** For a segment where ENBP is binding on 70% of policies, the elasticity model tells you: the profitable action would be a higher price, but the FCA prohibits it. The model's value here is identifying which customers in this segment are at lapse risk and might benefit from a retention discount — the opposite direction of the constraint.

**Task 3.** The retention objective sets every policy to its floor price (tech_prem × floor_loading = tech_prem at floor_loading=1.0). This maximises renewal probability at the cost of minimum margin. Mean predicted renewal prob will be higher than under the profit objective; expected profit per policy will be lower (or zero, since price = tech_prem means zero margin).
</details>

---

## Exercise 5: Sensitivity to confounder specification

**Reference:** Parts 3, 7

**What you will do:** Investigate how the ATE estimate changes when confounders are omitted, and understand what this implies about bias.

### Tasks

**Task 1.** Fit `RenewalElasticityEstimator(n_estimators=100, catboost_iterations=100)` three times:
- With the full confounder set: `["age", "ncd_years", "vehicle_group", "region", "channel"]`
- Without `channel`
- Without `channel` and `ncd_years`

Compare the ATEs and 95% CIs across the three specifications.

**Task 2.** In which direction does the ATE move as confounders are dropped? Is the pattern consistent with the confounding structure described in Part 3?

**Task 3.** The channel confounder is particularly important because PCW customers are both more expensive (PCW channel attracts higher-risk customers) and more elastic. Explain in one paragraph why omitting channel from the confounders biases the ATE, and in which direction.

<details>
<summary>Solution</summary>

**Task 2.** As confounders are dropped, the ATE typically becomes more negative — the bias from omitted variable confounding pulls the estimate in the same direction as the naive logistic regression. The more risk factors are controlled for, the less confounded the residual price variation.

**Task 3.** PCW customers face higher prices (because the channel selects for higher-risk profiles) and have higher elasticity (because PCW is a shopping-focused channel). Without channel in the confounder set, the DML nuisance model cannot account for the fact that PCW customers have higher log price changes *and* higher lapse rates for structural reasons. The residual D̃ for PCW customers carries a channel-correlated component that inflates the estimated price response. The ATE therefore overstates the true elasticity — it is more negative than the true causal effect.
</details>

---

## Exercise 6: The demand curve and commercial decision

**Reference:** Part 11

**What you will do:** Use `demand_curve()` to identify the profit-maximising price change and interpret the result for a commercial director.

### Tasks

**Task 1.** Run `demand_curve(est, df, price_range=(-0.20, 0.20, 40))`. Identify the profit-maximising price change and the associated renewal rate.

**Task 2.** The commercial director says: "The volume floor is 80% renewal rate. What is the maximum price increase consistent with that constraint?" Find the answer from the demand curve.

**Task 3.** Plot the demand curve. The chart should have dual axes: renewal rate on the left and expected profit per policy on the right. Mark the profit optimum with a vertical line. Write a two-sentence interpretation for a non-technical commercial director.

<details>
<summary>Solution</summary>

**Task 1.** The profit-maximising price change depends on the DGP parameters but will typically be in the range of +8% to +15%. At the profit optimum, the renewal rate will be somewhat below the current rate — the volume loss is exactly offset by the margin gain at the optimum.

**Task 2.** From the demand curve, find the row where `predicted_renewal_rate` is closest to 0.80 (80%). The corresponding `pct_price_change` is the maximum price increase consistent with the 80% floor. Subtract the current mean renewal rate from 0.80 to understand the implied volume loss.

**Task 3.** Interpretation: "A portfolio-wide price increase of approximately X% is expected to maximise total profit, generating £Y per policy on average. Accepting a smaller increase of Z% would maintain an 80% renewal rate while still generating above-current margins."
</details>

---

## Exercise 7: Validating the ENBP audit

**Reference:** Part 13

**What you will do:** Deliberately introduce an ENBP breach and confirm the audit catches it.

### Tasks

**Task 1.** Take the `priced_df` output from `RenewalPricingOptimiser.optimise()`. Manually overwrite the `optimal_price` for the first 10 rows to be ENBP + £50 (above the ceiling). Then run `opt.enbp_audit()` on this modified DataFrame. Does the audit correctly identify the 10 breaches?

```python
# Modify first 10 rows to breach ENBP
import numpy as np
prices = priced_df["optimal_price"].to_numpy().copy()
enbps  = priced_df["enbp"].to_numpy().copy()
prices[:10] = enbps[:10] + 50.0
priced_df_breached = priced_df.with_columns(pl.Series("optimal_price", prices))
```

**Task 2.** Inspect the `pct_above_enbp` column for the 10 breached policies. What is the mean percentage overcharge?

**Task 3.** In a production system, what should happen after the audit detects a breach? Write a short process description (3–5 steps) suitable for inclusion in a pricing governance document.

<details>
<summary>Solution</summary>

**Task 1.** Yes. The audit `filter(~pl.col("compliant"))` should return exactly 10 rows. `opt.enbp_audit()` will also raise a `UserWarning` listing the number of breaches.

**Task 2.** £50 above ENBP as a percentage of ENBP. For policies with ENBP around £500, this is approximately 10%. For policies with ENBP around £1,000, approximately 5%.

**Task 3.**
1. The pricing run is halted — no prices are issued to customers
2. The breach detail (policy IDs, offered price, ENBP, percentage above) is saved to a compliance incident log
3. The data quality team investigates the root cause: is the ENBP column incorrect, or did the optimiser produce the price in error?
4. The corrected pricing run is executed from scratch with the corrected data
5. The pricing actuary signs off the audit output from the corrected run before submission to the quoting system
</details>

---

## Exercise 8: End-to-end pipeline on a new book

**Reference:** All parts

**What you will do:** Run the complete pipeline — diagnostic, fit, GATE analysis, optimisation, audit — on a fresh dataset with different parameters.

### Setup

```python
# A smaller book with more price variation (easier identification)
df_ex8 = make_renewal_data(n=20_000, seed=99, price_variation_sd=0.12)
confounders_ex8 = ["age", "ncd_years", "vehicle_group", "channel"]
```

### Tasks

**Task 1.** Run the treatment variation diagnostic. Is identification sufficient? Report the variation fraction.

**Task 2.** Fit `RenewalElasticityEstimator(n_estimators=100, catboost_iterations=200, n_folds=3)`. On 20,000 records with 3 folds rather than 5, how does the CI width compare to the tutorial (50,000 records, 5 folds)?

**Task 3.** Compute the GATE by `channel` and compare to the tutorial's GATE by channel. Are the relative orderings (PCW most elastic, broker least) consistent across the two datasets?

**Task 4.** Run the per-policy optimisation and ENBP audit on `df_ex8`. Report: the mean optimal price, the fraction of ENBP-binding policies, and the breach count in the audit.

**Task 5.** Write the elasticity surface segment summary to a Polars DataFrame with columns `["ncd_years", "channel", "elasticity", "n"]`, sorted by elasticity ascending. Save it to `/tmp/elasticity_surface_ex8.csv`.

<details>
<summary>Solution</summary>

**Task 1.** With `price_variation_sd=0.12`, variation fraction will be higher than the tutorial (~0.65+). Identification is sufficient.

**Task 2.** The CI will be wider on 20,000 records vs. 50,000, and 3-fold cross-fitting gives noisier residuals than 5-fold. Expect CI width roughly 1.5–2× the tutorial CI.

**Task 3.** The relative ordering should be consistent: PCW most elastic, direct intermediate, broker least elastic. The absolute values will differ slightly due to sampling variation and the different seed.

**Task 4.** Breach count should be zero (the optimiser enforces the constraint). ENBP-binding fraction depends on the DGP parameters — expect 20–40% on this dataset.

**Task 5.**
```python
surface_ex8 = ElasticitySurface(est_ex8)
summary_ex8 = (
    surface_ex8.segment_summary(df_ex8, by=["ncd_years", "channel"])
    .select(["ncd_years", "channel", "elasticity", "n"])
    .sort("elasticity")
)
summary_ex8.write_csv("/tmp/elasticity_surface_ex8.csv")
print(summary_ex8)
```
</details>
