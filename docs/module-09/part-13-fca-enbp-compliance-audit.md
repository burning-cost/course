## Part 13: The ENBP compliance audit

Every renewal pricing run must be checked for PS21/5 compliance before prices are issued. The optimiser enforces the ENBP constraint as a hard ceiling, so breaches should not occur in its output — but the audit is the mechanism by which you demonstrate compliance to the FCA, not an assumption you make.

### Running the audit

```python
%md
## Part 13: ENBP compliance audit
```

```python
audit = opt.enbp_audit(priced_df)

n_breaches = int((~audit["compliant"]).sum())
print(f"ENBP audit results:")
print(f"  Total policies:  {len(audit):,}")
print(f"  Compliant:       {audit['compliant'].sum():,}")
print(f"  Breaches:        {n_breaches:,}")
print(f"  Breach rate:     {100 * n_breaches / len(audit):.3f}%")

if n_breaches == 0:
    print("\n  ALL POLICIES COMPLIANT with FCA ICOBS 6B.2")
else:
    print("\n  WARNING: Review breach detail before issuing prices")
    print(audit.filter(~pl.col("compliant")).head(20))
```

The audit checks FCA ICOBS 6B.2: for every policy, the offered renewal price must be at or below the ENBP for a new customer with identical risk characteristics on the same channel today. "On average" is not sufficient. The FCA can request a per-policy breakdown.

If you see breaches, the cause is one of: (1) the ENBP column in the data understates the true ENBP (data quality), (2) the technical premium floor exceeds the ENBP for some policies (rating inconsistency), or (3) a bug in the optimiser. Investigate before proceeding.

### Writing the audit trail to Unity Catalog

```python
# Production: save the audit trail with run metadata
from datetime import date

audit_with_meta = audit.with_columns([
    pl.lit(str(date.today())).alias("pricing_run_date"),
    pl.lit("v1.0").alias("model_version"),
    pl.lit("RenewalElasticityEstimator/CausalForestDML").alias("methodology"),
])

spark.createDataFrame(audit_with_meta.to_pandas()).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("pricing.motor.enbp_audit_log")

print("Audit trail written to pricing.motor.enbp_audit_log")
print(f"Schema: {audit_with_meta.columns}")
```

The audit table is the artefact you produce for a section 166 request. It should accumulate across pricing runs (append mode), allowing the FCA to see the history of compliance checking for any renewal cohort.

In a full production implementation, the audit row also includes the actuary's sign-off (a name and attestation timestamp stored separately), the git hash of the pricing code, and the MLflow run ID of the fitted elasticity model. These governance columns are the difference between a reproducible audit trail and a collection of CSV files.

### Understanding the margin distribution

```python
# Summarise the headroom to ENBP across the book
print("ENBP headroom distribution (£):")
print(audit.select("margin_to_enbp").describe())

# What fraction of the book has less than £10 headroom?
tight = (audit["margin_to_enbp"] < 10.0).sum()
print(f"\nPolicies with < £10 headroom: {tight:,} ({100 * tight / len(audit):.1f}%)")
```

Tight ENBP headroom across a large fraction of the book is a strategic signal: the FCA's pricing constraint, not customer demand, is the primary governor of your renewal book's profitability. In that environment, the value of the elasticity model shifts from "find the best price to charge" to "identify which customers are at lapse risk and worth a retention discount."
