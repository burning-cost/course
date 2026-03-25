## Part 5: Stage 1 — Configuration

All configurable values live in one cell. If a value appears in more than one place, it will eventually be inconsistent. This is not a style preference — it is the discipline that prevents the configuration drift that makes pipelines unreproducible.

Add a markdown cell:

```python
%md
## Stage 1: Configuration
```

Then add this code cell:

```python
import json
from datetime import date

# -----------------------------------------------------------------------
# Unity Catalog coordinates
# -----------------------------------------------------------------------
CATALOG = "main"          # change to "hive_metastore" on Databricks Free Edition
SCHEMA  = "motor_q2_2026" # name by review cycle, not model version

TABLES = {
    "raw":                 f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":            f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":    f"{CATALOG}.{SCHEMA}.freq_predictions",
    "freq_relativities":   f"{CATALOG}.{SCHEMA}.freq_relativities",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "rate_change":         f"{CATALOG}.{SCHEMA}.rate_action_factors",
    "efficient_frontier":  f"{CATALOG}.{SCHEMA}.efficient_frontier",
    "pipeline_audit":      f"{CATALOG}.{SCHEMA}.pipeline_audit",
}

# -----------------------------------------------------------------------
# Pipeline parameters
# -----------------------------------------------------------------------
N_POLICIES      = 200_000
N_OPTUNA_TRIALS = 20      # increase to 40 for production runs
LR_TARGET       = 0.72    # loss ratio target
VOLUME_FLOOR    = 0.97    # minimum retention fraction
FACTOR_LOWER    = 0.90    # lower bound on each factor adjustment
FACTOR_UPPER    = 1.15    # upper bound on each factor adjustment
CONFORMAL_ALPHA = 0.10    # 1 - 0.10 = 90% prediction intervals

# -----------------------------------------------------------------------
# Run identification
# -----------------------------------------------------------------------
RUN_DATE = str(date.today())

print(f"Run date:           {RUN_DATE}")
print(f"Catalog / schema:   {CATALOG}.{SCHEMA}")
print(f"LR target:          {LR_TARGET:.0%}")
print(f"Volume floor:       {VOLUME_FLOOR:.0%}")
print(f"Conformal alpha:    {CONFORMAL_ALPHA}")
print(f"Optuna trials:      {N_OPTUNA_TRIALS}")
print("\nTable names:")
for k, v in TABLES.items():
    print(f"  {k:<25} {v}")
```

### Create the schema

```python
if CATALOG != "hive_metastore":
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

mlflow.set_experiment(
    f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/motor-pipeline-m08"
)

print(f"Schema {CATALOG}.{SCHEMA} ready.")
```

### Why the table dictionary matters

Any notebook that names a table twice will eventually have those two names disagree. A table dictionary eliminates that failure mode: every stage reads its table name from `TABLES["key"]`, never from a string literal.

The pattern also makes refactoring safe. If you rename the features table from `features` to `engineered_features`, you change one value in the dictionary. Every downstream stage that reads `TABLES["features"]` picks up the new name automatically. If you had used string literals, you would need to search and replace across the entire notebook — and you would miss at least one.

### Setting the MLflow experiment

`mlflow.set_experiment()` creates a named experiment folder under your Databricks user path if it does not already exist. All MLflow runs in this pipeline will appear under that experiment. The experiment name should match your review cycle so you can find it months later:

```
/Users/your.email/motor-pipeline-m08
/Users/your.email/motor-pipeline-q3-2026
```

If you run this notebook multiple times in the same session, subsequent runs will add new entries to the same experiment. Each run has a unique `run_id` that the audit record captures.
