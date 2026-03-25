## Part 15: Practical deployment considerations

### How often to refit

The elasticity model captures a structural relationship: how customers respond to price changes at the current state of the market. That relationship changes — slowly, but meaningfully — with market conditions, competitor behaviour, PCW algorithmic changes, and the mix of customers acquired.

Our recommendation for a medium-sized UK motor book:

- **Renewal elasticity model**: annual refit, triggered early by major pricing events (bulk re-rate, entry of a new aggregator, post-FCA review). The refitting requires sufficient residual treatment variation, which accumulates from rate reviews — weekly refitting gains nothing.
- **Treatment variation diagnostic**: run before every fit. If variation drops below the threshold between refits, the model should not be used until the cause is understood.
- **ENBP audit**: run after every pricing file generation, not just after model refits.

### What you need to store at quote time

The most common data quality problem in insurance demand modelling is not having the technical premium stored at the moment the quote was issued.

The technical premium used in the DML treatment variable must be the value from the underwriting system at the time of quoting. A retrospectively recalculated technical premium — even from the same model — introduces errors because the model may have changed, the reference data (e.g. claims development) will have changed, and the quote-date effect is lost.

If your renewal table does not contain `tech_prem_at_quote` or equivalent, your treatment variable is log(offer_price / last_premium) rather than log(offer_price / tech_prem). This is less clean causally — last premium is correlated with the previous year's risk profile rather than this year's — but is still workable if the book is re-rated consistently.

The fix is a one-time schema change: add a `tech_prem_at_quote` column to the renewal event table. It costs nothing computationally and permanently improves identification.

### Production data pipeline

In production, the full pipeline looks like:

```python
# 1. Load renewal data from Delta (replace 'pricing.motor.renewals_2025_q1')
df_prod = pl.from_pandas(
    spark.table("pricing.motor.renewals_2025_q1").toPandas()
)

# 2. Run diagnostic
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(df_prod, confounders=confounders)
if report.weak_treatment:
    raise RuntimeError(f"Weak treatment: {report.summary()}")

# 3. Fit estimator
est = RenewalElasticityEstimator(n_estimators=200, catboost_iterations=500)
est.fit(df_prod, confounders=confounders)

# 4. Optimise per-policy
opt = RenewalPricingOptimiser(est, technical_premium_col="tech_prem", enbp_col="enbp")
priced = opt.optimise(df_prod, objective="profit")

# 5. Audit
audit = opt.enbp_audit(priced)
if (~audit["compliant"]).any():
    raise RuntimeError("ENBP breach detected — do not issue prices")

# 6. Write to Delta
spark.createDataFrame(priced.to_pandas()).write.format("delta") \
    .mode("overwrite").saveAsTable("pricing.motor.renewal_prices_2025_q1")
spark.createDataFrame(audit.to_pandas()).write.format("delta") \
    .mode("append").saveAsTable("pricing.motor.enbp_audit_log")
```

Steps 2 and 6 are guardrails: the diagnostic raises before the fit if the data is not identifiable, and the audit raises before writing if prices breach ENBP. Both are implemented as hard failures rather than warnings, because a silent compliance failure is worse than a noisy one.

### Limitations of the linear demand approximation

The per-policy optimiser uses a linear approximation to the demand curve. For price changes beyond ±20%, the linear approximation diverges meaningfully from a logistic specification. In practice, post-PS21/5 renewal price changes are constrained by ENBP — which typically limits changes to ±15% — so the approximation error is small.

The more significant limitation is the per-row observed renewal indicator as the P₀ baseline. Each customer's observed renewal is a single binary outcome, not a probability. The optimiser smooths it toward the portfolio mean (20% weight), but noisy baselines do reduce optimisation precision for customers at the extremes of the renewal distribution. With enough data, a neighbourhood estimate of P₀ (e.g., k-nearest-neighbours renewal rate in feature space) would improve this.
