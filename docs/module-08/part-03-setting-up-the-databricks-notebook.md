## Part 3: Setting up the Databricks notebook

### Creating the notebook

Log into your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your personal folder under **Users > your-email**. Click **+** and choose **Notebook**. Name it `module-08-end-to-end-pipeline`. Leave the language as Python. Click **Create**.

The notebook opens with one empty cell. Click **Connect** in the top-right and select your running cluster. If no cluster is running, go to **Compute**, find your cluster, and click **Start**. Allow 3-5 minutes for startup. Do not run any cells until the cluster shows a green status icon.

### Stage 0: Installing the libraries

The first cell installs all required libraries. `%pip install` on Databricks installs packages into the running Python environment on the cluster driver. The `dbutils.library.restartPython()` call restarts the Python interpreter so newly installed packages are importable. This step is required because Databricks caches the Python environment state before `%pip install` runs — without the restart, import statements will still target the pre-install state.

In the first cell, paste this and run it with Shift+Enter:

```python
%pip install \
    catboost \
    optuna \
    polars \
    "insurance-conformal[catboost]" \
    "insurance-monitoring[mlflow]" \
    insurance-optimise \
    shap-relativities \
    --quiet
```

Wait for the installation output to complete. The final lines will confirm the installed versions. Then, in a new cell:

```python
dbutils.library.restartPython()
```

After the restart, verify the imports in a new cell:

```python
import numpy as np
import polars as pl
import mlflow
import optuna
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor
from insurance_monitoring.calibration import CalibrationChecker
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier
from shap_relativities import SHAPRelativities

print(f"Polars:  {pl.__version__}")
print(f"MLflow:  {mlflow.__version__}")
print(f"Optuna:  {optuna.__version__}")
print("All libraries imported successfully.")
```

If you see `ModuleNotFoundError` for any library, run the `%pip install` cell again and restart again. Do not try to import before the restart completes.

### Local development

For running outside Databricks:

```bash
uv add catboost optuna polars "insurance-conformal[catboost]" \
       "insurance-monitoring[mlflow]" insurance-optimise shap-relativities
```

The `spark` object and `dbutils` are not available outside Databricks. Wrap any Spark calls in a guard:

```python
try:
    spark
    IN_DATABRICKS = True
except NameError:
    IN_DATABRICKS = False
```

The Delta table writes and MLflow experiment setup are Databricks-specific. The model training, SHAP extraction, conformal calibration, and rate optimisation all run identically in a local environment.

### Unity Catalog: why the three-part naming matters

The pipeline writes every output to Unity Catalog Delta tables. Table names use three-part notation: `catalog.schema.table`. The catalog is the governance boundary — different teams or lines of business can have separate catalogs with independent access controls. The schema groups tables by review cycle.

Name schemas by review cycle, not by model version:

```
motor_q2_2026    <- correct: tells you when and what
motor_v3         <- wrong: tells you nothing about when
```

On Databricks Free Edition, you may only have access to `hive_metastore`. In that case, tables are two-part (`schema.table`) and the `CREATE CATALOG` statement is not available. Set `CATALOG = "hive_metastore"` in Stage 1. The rest of the pipeline runs identically.

Unity Catalog logs every read and write to every table. Combined with Delta's version history — every overwrite increments the version counter, and you can read any historical version with `.option("versionAsOf", N)` — you have complete data provenance for the pipeline audit record.
