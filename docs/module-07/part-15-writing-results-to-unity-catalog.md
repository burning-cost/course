## Part 15: Writing results to Unity Catalog

The factor adjustments table and frontier are the production artefacts. They go to the data team for rating engine implementation, to the pricing committee for sign-off, and into the FCA audit trail.

```python
from datetime import date

CATALOG = "pricing"
SCHEMA  = "motor"
RUN_DATE = str(date.today())

# factor_adj holds the solved multipliers (assigned in Part 13)
# Define here in case you are running Part 15 independently
factor_adj = result.factor_adjustments

# --- Factor adjustments table ---
adj_records = [
    {
        "run_date":         RUN_DATE,
        "factor_name":      fname,
        "adjustment":       float(factor_adj.get(fname, 1.0)),
        "pct_change":       float((factor_adj.get(fname, 1.0) - 1) * 100),
        "lr_target":        LR_TARGET,
        "volume_floor":     VOLUME_FLOOR,
        "factor_lower_cap": FACTOR_LOWER,
        "factor_upper_cap": FACTOR_UPPER,
        "expected_lr":      float(result.expected_loss_ratio),
        "expected_volume":  float(result.expected_volume_ratio),
        "converged":        bool(result.converged),
        "objective_value":  float(result.objective_value),
    }
    for fname in FACTOR_NAMES
]

spark.createDataFrame(adj_records) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_factors")

# --- Efficient frontier table ---
spark.createDataFrame(frontier_df) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.efficient_frontier")

print(f"Written to {CATALOG}.{SCHEMA}.rate_action_factors ({len(adj_records)} factors)")
print(f"Written to {CATALOG}.{SCHEMA}.efficient_frontier ({len(frontier_df)} rows)")
```

Both tables carry `run_date`, `lr_target`, `volume_floor`, and the factor caps. This is enough information to reconstruct the optimisation problem from the outputs alone. Delta's versioning means the exact data used to produce each rate action is frozen in history — essential for the FCA audit trail.

**What goes in the pricing committee pack:**

1. The factor adjustments table (the six values from `result.factor_adjustments`)
2. The efficient frontier plot (the two-panel chart from Part 10)
3. The ENBP compliance statement (zero violations, verified per-policy)
4. The premium impact distribution (mean, median, 10th-90th percentile)
5. The convergence confirmation (`result.converged = True`)
6. The constraint binding summary (which constraints were active at the optimum)