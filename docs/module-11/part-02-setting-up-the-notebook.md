## Part 2: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder (or your Git repo if you have one connected).

Click **+** and choose **Notebook**. Name it `module-11-model-monitoring`. Keep the default language as Python. Click **Create**.

Connect it to a running cluster. Click the cluster selector at the top of the notebook. If it shows "Detached," click it and select your cluster. Once the cluster name appears in green, you are connected.

### Install the libraries

In the first cell, run:

```python
%pip install insurance-monitoring insurance-datasets catboost polars mlflow
```

Wait for the install to complete, then restart the Python session:

```python
dbutils.library.restartPython()
```

Here is what each library does in this module:

- **insurance-monitoring** - the monitoring library we use throughout. Provides `PSICalculator`, `CSICalculator`, `AERatio`, `GiniDrift`, and `MonitoringReport`.
- **insurance-datasets** - the synthetic UK motor portfolio. We use `load_motor()` to get consistent data across all modules.
- **catboost** - we load the trained model from Module 8 and run predictions against current data.
- **polars** - our data manipulation library. Fast, memory-efficient, and explicit about types.
- **mlflow** - we load the registered model from the MLflow Model Registry and log monitoring results.

### Confirm the imports

In the next cell:

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from insurance_datasets import load_motor
from insurance_monitoring import (
    PSICalculator,
    CSICalculator,
    AERatio,
    GiniDrift,
    MonitoringReport,
)

print(f"Polars:              {pl.__version__}")
print(f"insurance-monitoring imported OK")
print(f"insurance-datasets:  load_motor available")
print("All imports OK")
```

You should see version numbers with no errors. If `ModuleNotFoundError` appears for `insurance-monitoring`, run the `%pip install` cell again and make sure you ran `dbutils.library.restartPython()` afterwards.

### Configuration cell

All configurable values go in one cell. If a value appears in more than one place, it will eventually be inconsistent:

```python
from datetime import date

# -----------------------------------------------------------------------
# Unity Catalog coordinates
# -----------------------------------------------------------------------
# In Databricks Free Edition, your catalog is likely "main".
# Change this to match your workspace.
CATALOG = "main"
SCHEMA  = "motor_monitoring"

TABLES = {
    "reference_data":  f"{CATALOG}.{SCHEMA}.reference_data",
    "current_data":    f"{CATALOG}.{SCHEMA}.current_data",
    "monitoring_log":  f"{CATALOG}.{SCHEMA}.monitoring_log",
    "psi_results":     f"{CATALOG}.{SCHEMA}.psi_results",
    "csi_results":     f"{CATALOG}.{SCHEMA}.csi_results",
    "ae_results":      f"{CATALOG}.{SCHEMA}.ae_results",
    "gini_results":    f"{CATALOG}.{SCHEMA}.gini_results",
}

# -----------------------------------------------------------------------
# Monitoring parameters
# -----------------------------------------------------------------------
# REFERENCE_DATE: when the model was trained / last validated.
# CURRENT_DATE:   the end of the monitoring window we are assessing.
# In production, CURRENT_DATE = str(date.today()).
REFERENCE_DATE  = "2023-12-31"
CURRENT_DATE    = "2024-06-30"

# Model name in the MLflow Model Registry (from Module 8)
MODEL_NAME      = "motor_frequency_catboost"
MODEL_VERSION   = "1"

# PSI/CSI bin count. 10 is the standard; reduce to 5 for sparse features.
N_BINS = 10

# -----------------------------------------------------------------------
# Run identification
# -----------------------------------------------------------------------
RUN_DATE = str(date.today())

print(f"Run date:          {RUN_DATE}")
print(f"Reference period:  {REFERENCE_DATE}")
print(f"Current period:    {CURRENT_DATE}")
print(f"Catalog/schema:    {CATALOG}.{SCHEMA}")
print(f"Model:             {MODEL_NAME} v{MODEL_VERSION}")
```

### Create the schema

```python
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready.")
```

If you are on `hive_metastore` rather than Unity Catalog, remove `{CATALOG}.` from the `CREATE SCHEMA` statement and use two-part table names throughout.

### Load the data

We need two time windows: the reference window (the data the model was trained and validated on) and the current window (recent live policies). We use `load_motor()` and split by policy start date:

```python
# Load the full motor dataset
df = load_motor()
print(f"Total records: {df.shape[0]:,}")
print(df.dtypes)
```

`load_motor()` returns a Polars DataFrame with UK motor policies. The schema includes `policy_start_date`, `claim_count`, `exposure`, `driver_age`, `vehicle_age`, `vehicle_group`, `region`, and several other features. We will use this schema throughout the module.

In the next cell, split into reference and current:

```python
df_reference = df.filter(
    pl.col("policy_start_date") <= pl.lit(REFERENCE_DATE).str.to_date()
)

df_current = df.filter(
    (pl.col("policy_start_date") > pl.lit(REFERENCE_DATE).str.to_date()) &
    (pl.col("policy_start_date") <= pl.lit(CURRENT_DATE).str.to_date())
)

print(f"Reference records: {df_reference.shape[0]:,}")
print(f"Current records:   {df_current.shape[0]:,}")
```

You should see roughly 70-80% of records in the reference set and 20-30% in the current set, depending on the date distribution in the synthetic dataset. If the current set is empty, check that `CURRENT_DATE` is after `REFERENCE_DATE` and that the dataset contains records in that range.

We now have everything we need. The next part explains what model drift actually is before we start measuring it.
