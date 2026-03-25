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

- **insurance-monitoring** — the monitoring library we use throughout. Provides function-level APIs (`psi`, `csi`, `ae_ratio_ci`, `gini_coefficient`, `gini_drift_test`) and the `MonitoringReport` dataclass.
- **insurance-datasets** — the synthetic UK motor portfolio. We use `load_motor()` to get consistent data across all modules.
- **catboost** — we load the trained model from Module 8 and run predictions against current data.
- **polars** — our data manipulation library. Fast, memory-efficient, and explicit about types.
- **mlflow** — we load the registered model from the MLflow Model Registry and log monitoring results.

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

from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi, csi
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

print(f"Polars:              {pl.__version__}")
print("insurance-monitoring imported OK")
print("insurance-datasets:  load_motor available")
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
CATALOG = "main"
SCHEMA  = "motor_monitoring"

TABLES = {
    "monitoring_log": f"{CATALOG}.{SCHEMA}.monitoring_log",
    "csi_results":    f"{CATALOG}.{SCHEMA}.csi_results",
    "ae_results":     f"{CATALOG}.{SCHEMA}.ae_results",
}

# -----------------------------------------------------------------------
# Monitoring parameters
# -----------------------------------------------------------------------
REFERENCE_YEAR = 2022  # policies with inception_year <= this are "reference"
CURRENT_YEAR   = 2023  # policies with inception_year == this are "current"

MODEL_NAME    = "motor_frequency_catboost"
MODEL_VERSION = "1"

N_BINS = 10  # PSI/CSI bin count. 10 is standard; reduce to 5 for sparse features.

RUN_DATE = str(date.today())

print(f"Run date:          {RUN_DATE}")
print(f"Reference period:  inception_year <= {REFERENCE_YEAR}")
print(f"Current period:    inception_year == {CURRENT_YEAR}")
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

We need two time windows: the reference window (the data the model was trained and validated on) and the current window (recent live policies). We use `load_motor(polars=True)` to get a Polars DataFrame and split by inception year:

```python
df = load_motor(polars=True)
print(f"Total records: {df.shape[0]:,}")
print(f"Columns: {df.columns}")

df_reference = df.filter(pl.col("inception_year") <= REFERENCE_YEAR)

df_current = df.filter(pl.col("inception_year") == CURRENT_YEAR)

print(f"Reference records: {df_reference.shape[0]:,}")
print(f"Current records:   {df_current.shape[0]:,}")
```

`load_motor()` returns a pandas DataFrame by default; pass `polars=True` to get a Polars DataFrame directly. The dataset uses `inception_year` (integer, 2019–2023) as the cohort identifier — there is no `policy_start_date` column. Split on `inception_year` to get stable reference and current windows. You should see roughly 80% of records in the reference set and 20% in the current set.

We now have everything we need. The next part explains what model drift actually is before we start measuring it.
