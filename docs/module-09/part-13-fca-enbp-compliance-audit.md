## Part 13: FCA ENBP compliance audit

Every renewal pricing run must be audited for PS21/5 compliance before the prices are issued to customers. The `enbp_audit()` method produces the per-policy compliance report.

```python
%md
## Part 13: ENBP compliance audit
```

```python
audit = opt.enbp_audit(priced_df)

n_breaches = int((audit["compliant"] == False).sum())
print(f"ENBP audit results:")
print(f"Total policies:    {len(audit):,}")
print(f"Compliant:         {(audit['compliant']).sum():,}")
print(f"Breaches:          {n_breaches:,}")
print(f"Breach rate:       {100 * n_breaches / len(audit):.2f}%")

if n_breaches == 0:
    print("\nALL POLICIES COMPLIANT with FCA ICOBS 6B.2")
else:
    print("\nWARNING: Review breach detail before issuing prices")
    print(audit.filter(pl.col("compliant") == False).head(10))
```

The optimiser enforces the ENBP constraint as a hard ceiling, so there should be zero breaches in the output. Any breach would indicate a bug in the optimiser or a data quality problem (for example, an ENBP value lower than the technical premium floor).

### Understanding what the audit is checking

The ENBP audit checks FCA ICOBS 6B.2, which states: "A firm must not offer a renewal price which is higher than the equivalent new business price it would offer to a retail customer with the same relevant characteristics."

The `enbp` column in the renewal dataset represents what a new customer with identical risk characteristics would be quoted today on the same channel. The audit confirms that for every single policy, the offered renewal price is at or below this.

"On average" is not sufficient. The FCA can ask for a per-policy breakdown. The `enbp_audit()` output is designed to be saved to a Delta table as evidence.

### Saving the audit trail to Unity Catalog

In production, you would write the audit output to a Unity Catalog table:

```python
# In production - save audit trail
# Replace with your actual catalog and schema names

# audit_pd = audit.to_pandas()
# spark.createDataFrame(audit_pd).write.format("delta").mode("append").saveAsTable(
#     "pricing.motor.enbp_audit_log"
# )

# For now, confirm the columns that would be saved
print("Audit table schema:")
for col in audit.columns:
    print(f"  {col}: {audit[col].dtype}")
```

The complete audit trail should include the run date, the actuary who signed off, the version of the demand model used, and the ENBP source. These are governance additions on top of the per-policy data.

### Using the insurance-optimise demand compliance tools

The `insurance_optimise.demand` submodule provides additional compliance utilities through its `ENBPChecker` class. This is a higher-level wrapper that produces summary reports useful for compliance officers rather than actuaries:

```python
from insurance_optimise.demand.compliance import ENBPChecker

# Convert the renewal data to the format insurance_optimise.demand expects
# (it uses 'renewal_price' and 'nb_equivalent_price' column names)
df_renewals_for_checker = df_renewals.rename({
    "enbp": "nb_equivalent_price",
}).with_columns(
    (pl.col("last_premium") * pl.col("log_price_change").exp()).alias("renewal_price")
)

checker = ENBPChecker(tolerance=0.0)
try:
    compliance_report = checker.check(df_renewals_for_checker)
    print(f"Breaches detected: {compliance_report.n_breaches}")
    print(f"By channel: {compliance_report.by_channel}")
except (KeyError, ValueError) as e:
    # Column name mismatch -- the schema requirements for ENBPChecker are strict.
    # This section may produce a schema error depending on the version of insurance-optimise;
    # if so, adjust column names as shown below. Do not rely on silent failure here --
    # a compliance tool that fails quietly is worse than one that fails loudly.
    print(f"Schema error (adjust column names): {e}")
    print("Required columns:", ["policy_id", "renewal_price", "nb_equivalent_price",
                                  "lapsed", "tenure_years"])
```