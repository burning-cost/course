## Part 6: Fitting the severity GLM

### What is different about severity

Severity GLMs differ from frequency GLMs in three important ways:

1. **We fit on claimed policies only.** A policy with zero claims contributes nothing to the severity distribution. Fitting severity on all policies (including the zeros) would be mathematically wrong.

2. **The response variable is average severity per claim, not total incurred.** Total incurred mixes severity and claim count. We want to model the cost per event, not the cost per policy.

3. **We use claim count as a variance weight, not an exposure offset.** A policy with 3 claims gives us an average of 3 severities, which is more informative than an average of 1. The `var_weights` argument tells statsmodels to weight each observation's contribution by its claim count.

### NCD does not belong in the severity model

NCD years are excluded from the severity formula and this is a deliberate modelling decision, not an oversight.

NCD reflects driving behaviour and correlates with claim frequency - drivers with zero NCD have more accidents. But conditional on a claim occurring, the claim cost is not systematically different between NCD=0 and NCD=5 drivers. Including NCD in the severity model would capture frequency effects through the back door, double-counting the NCD signal. The frequency model picks it up correctly; the severity model should not.

This is the kind of thing Emblem users sometimes handle by trial and error. In Python, the decision is explicit in the formula string.

### Large loss truncation

Before fitting the severity GLM on real motor data, you need to decide what to do about large personal injury claims. Bodily injury claims on UK motor books are typically 10-100x the average accidental damage claim. An untruncated Gamma severity model will be driven by whichever risk characteristics correlate with the handful of catastrophic PI claims in your portfolio.

Standard practice is to cap large losses at £100k-£250k and model the excess separately, or to separate PI claims from property damage. For the synthetic data here we have no large PI exposure - the severity DGP is a simple Gamma with mean £3,500. On real data, always add this step before the severity GLM:

```python
LARGE_LOSS_THRESHOLD = 100_000  # £100k cap - adjust for your book

df_sev_all = df.with_columns(
    pl.col("incurred").clip(upper_bound=LARGE_LOSS_THRESHOLD * pl.col("claim_count"))
    .alias("incurred_capped")
)
n_capped = df_sev_all.filter(
    pl.col("incurred") > LARGE_LOSS_THRESHOLD * pl.col("claim_count")
).shape[0]
print(f"Policies capped at £{LARGE_LOSS_THRESHOLD/1000:.0f}k: {n_capped}")
```

Document the cap in your model log. It is a modelling assumption that materially affects severity relativities, and anyone trying to reproduce your results without knowing the cap will not match your numbers.

### Fitting the model

```python
# Severity data: claimed policies only
df_sev = df.filter(pl.col("claim_count") > 0)
df_sev = df_sev.with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

print(f"Severity model: {len(df_sev):,} claimed policies")
print(f"Mean average severity: £{df_sev['avg_severity'].mean():,.0f}")

df_sev_pd = df_sev.to_pandas()

sev_formula = (
    "avg_severity ~ "
    "C(area) + "
    "vehicle_group"
    # NCD deliberately excluded - see above
)

glm_sev = smf.glm(
    formula=sev_formula,
    data=df_sev_pd,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_sev_pd["claim_count"],
).fit()

print(f"\nSeverity GLM:")
print(f"Converged: {glm_sev.converged}")
print(f"Gamma scale (phi): {glm_sev.scale:.4f}")
print(f"CV of severity: {np.sqrt(glm_sev.scale):.3f}")
```

**What to check:**

- `Converged: True` - as with the frequency model
- Gamma scale (phi): this is the dispersion parameter for the Gamma family. A coefficient of variation (CV = sqrt(phi)) of 0.5-0.8 is typical for motor accidental damage severity. If it is below 0.3, check whether your data has already been capped. If it is above 1.5, you may have a mixture of claim types (small property damage plus large PI) that would be better separated.

Extract severity relativities:

```python
sev_rels = extract_freq_relativities(
    glm_sev,
    base_levels={"area": "A"},
)

print("Severity area relativities:")
print(sev_rels.filter(pl.col("feature") == "area"))

# The true severity DGP has no area effect. The severity area relativities
# should all be close to 1.0 (within sampling noise).
print("\nIf the model is correct, all area relativities should be close to 1.0")
print("because area was not in the severity data-generating process.")
```

This is an important check. The synthetic data was generated with no area effect on severity. So the severity area relativities should be statistically indistinguishable from 1.0. If they are not, something is wrong with the model specification.