## Part 5: Stage 1 -- Configuration

All configurable values live in one cell. This is the first rule of pipeline design. If a value appears in more than one place, it will eventually be inconsistent.

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
# CATALOG: the top-level container. In Databricks Free Edition, you may
# only have access to a catalog called "main" or "hive_metastore".
# Change this to match your Databricks environment.
CATALOG = "main"

# SCHEMA: the database within the catalog. Name it by review cycle, not
# version number. "motor_q2_2026" tells you exactly when and what.
# "motor_v3" tells you nothing about when it was run.
SCHEMA  = "motor_q2_2026"

# TABLES: all table names in one dictionary. Change a name here; it
# changes everywhere. Never hard-code a table name twice.
TABLES = {
    "raw":                 f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":            f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":    f"{CATALOG}.{SCHEMA}.freq_predictions",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "rate_change":         f"{CATALOG}.{SCHEMA}.rate_action_factors",
    "efficient_frontier":  f"{CATALOG}.{SCHEMA}.efficient_frontier",
    "pipeline_audit":      f"{CATALOG}.{SCHEMA}.pipeline_audit",
}

# -----------------------------------------------------------------------
# Model and pipeline parameters
# -----------------------------------------------------------------------
LR_TARGET       = 0.72    # Loss ratio target for rate optimisation
VOLUME_FLOOR    = 0.97    # Minimum portfolio volume retention
CONFORMAL_ALPHA = 0.10    # 1 - alpha = 90% coverage intervals
N_OPTUNA_TRIALS = 20      # Increase to 40 for production runs

# -----------------------------------------------------------------------
# Run identification
# -----------------------------------------------------------------------
RUN_DATE = str(date.today())

print(f"Run date:           {RUN_DATE}")
print(f"Catalog / schema:   {CATALOG}.{SCHEMA}")
print(f"Loss ratio target:  {LR_TARGET:.0%}")
print(f"Conformal alpha:    {CONFORMAL_ALPHA}")
print(f"Optuna trials:      {N_OPTUNA_TRIALS}")
print("\nTable names:")
for k, v in TABLES.items():
    print(f"  {k:<25} {v}")
```

**What you should see:** A clean print of all configuration values. If any table name looks wrong, fix it here before running any subsequent stage.

### A note on Unity Catalog

Unity Catalog is Databricks' governance layer for data assets. It uses a three-part naming convention: `catalog.schema.table`. The catalog is the top-level container -- in an enterprise environment it might be called `pricing` or `insurance`. The schema is a logical grouping within the catalog -- we name it by review cycle. The table is the individual data asset.

In Databricks Free Edition, you may only have access to a legacy metastore called `hive_metastore`. In that case, your table names are two-part: `schema.table`. Change `CATALOG = "main"` to `CATALOG = "hive_metastore"` and adjust the TABLES dictionary accordingly. The rest of the pipeline runs identically either way.

The reason we use Unity Catalog for pricing is audit. Unity Catalog logs every read and write to every table. Combined with Delta's version history, you can reconstruct what data was read by any pipeline run on any date in the past. For Consumer Duty compliance, this is not optional -- you need to be able to demonstrate, three years later, what data informed a specific pricing decision.

### Create the schema

The schema must exist before you can write to it. Run this in a new cell:

```python
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready.")
```

If you are using `hive_metastore`, omit the `CREATE CATALOG` line -- the legacy metastore does not use catalogs.