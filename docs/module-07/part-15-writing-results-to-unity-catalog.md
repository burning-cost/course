## Part 15: Writing results to Unity Catalog

The per-policy optimal premiums, the audit trail, and the frontier data are the production artefacts. They go to the data team for rating engine implementation, to the pricing committee for sign-off, and into the FCA audit trail.

```python
import polars as pl
from datetime import date

CATALOG  = "pricing"
SCHEMA   = "motor"
RUN_DATE = str(date.today())

# --- Per-policy optimal premiums ---
policy_records = (
    df
    .with_columns([
        pl.Series("optimal_multiplier", result.multipliers.tolist()),
        pl.Series("optimal_premium",    result.new_premiums.tolist()),
        pl.Series("expected_demand",    result.expected_demand.tolist()),
        pl.Series("rate_change_pct",    result.summary_df["rate_change_pct"].to_list()),
        pl.Series("enbp_binding",       result.summary_df["enbp_binding"].to_list()),
        pl.lit(RUN_DATE).alias("run_date"),
        pl.lit(LR_TARGET).alias("lr_target"),
        pl.lit(RETENTION_FLOOR).alias("retention_floor"),
        pl.lit(MAX_RATE_CHANGE).alias("max_rate_change"),
    ])
)

# --- Efficient frontier ---
frontier_records = frontier_result.data.with_columns(
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(LR_TARGET).alias("lr_target"),
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

(
    spark.createDataFrame(policy_records.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_policies")
)
print(f"Policy premiums written to {CATALOG}.{SCHEMA}.rate_action_policies ({len(policy_records):,} rows)")

(
    spark.createDataFrame(frontier_records.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.efficient_frontier")
)
print(f"Frontier data written to {CATALOG}.{SCHEMA}.efficient_frontier ({len(frontier_records)} rows)")
```

Both tables carry `run_date`, `lr_target`, and `retention_floor`. This is enough information to reconstruct the optimisation problem from the outputs alone. Delta's versioning means the exact data used to produce each rate action is frozen in history — essential for the FCA audit trail.

### The JSON audit trail

The `result.audit_trail` dict is a complete record of the optimisation: inputs, constraints, solver settings, solution summary, shadow prices, and convergence info. Save it to a file or to a Unity Catalog volume:

```python
import json

# Save to DBFS
audit_path = f"/dbfs/pricing/motor/rate_action_{RUN_DATE}_audit.json"
result.save_audit(audit_path)
print(f"Audit trail saved to: {audit_path}")

# Or write the JSON string to a Delta table row
audit_json = result.to_json()
spark.createDataFrame([{
    "run_date": RUN_DATE,
    "audit_json": audit_json,
    "converged": bool(result.converged),
    "expected_profit": float(result.expected_profit),
    "expected_lr": float(result.expected_loss_ratio),
    "expected_retention": float(result.expected_retention),
}]).write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_audit_log")
```

**What goes in the pricing committee pack:**

1. The efficient frontier plot (from Part 10): profit vs retention, mark the chosen operating point
2. The distribution of rate changes: mean, median, 10th–90th percentile
3. The ENBP compliance statement: zero violations per-policy
4. The shadow prices: which constraints are binding and their marginal cost
5. The convergence confirmation: `result.converged = True`, `result.n_iter` iterations
6. The stochastic sensitivity: how much higher the rates need to be for 90% confidence

The audit trail JSON satisfies FCA requirements under Consumer Duty for documenting the basis of pricing decisions.
