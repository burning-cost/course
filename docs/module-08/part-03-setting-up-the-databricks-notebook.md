## Part 3: Setting up the Databricks notebook

### Creating the notebook

Log into your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your personal folder (usually listed under **Workspace > Users > your-email**).

Click the **+** button and choose **Notebook**. Name it `module-08-end-to-end-pipeline`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Connect it to your cluster by clicking **Connect** in the top-right and selecting your running cluster. If no cluster is running, go to **Compute** in the left sidebar, find your cluster, and click **Start**. Cluster startup takes 3-5 minutes.

Do not run any cells until the cluster shows a green status icon.

### Stage 0: Installing the libraries

The first cell of every notebook in this course installs all required libraries. On Databricks, `%pip install` installs packages into the running Python environment on the cluster. The `dbutils.library.restartPython()` call immediately after restarts the Python interpreter so the newly installed packages are importable.

In the first cell, type this and run it with Shift+Enter:

```python
%pip install \
    catboost \
    optuna \
    mlflow \
    polars \
    "insurance-cv" \
    "insurance-conformal[catboost]" \
    "rate-optimiser" \
    insurance-datasets \
    --quiet
```

You will see pip installation output scrolling for 30-90 seconds. Wait for it to complete. The last line will say something like `Successfully installed catboost-1.x.x ...`.

Once the install finishes, run this in a new cell:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables you defined before the restart are gone -- that is expected. The install cell must always be the first cell, and you must run all subsequent cells after the restart.

After the restart, in a new cell, confirm everything imported:

```python
import polars as pl
import numpy as np
import mlflow
import optuna
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor
from insurance_datasets import load_motor

print(f"Polars:   {pl.__version__}")
print(f"NumPy:    {np.__version__}")
print(f"MLflow:   {mlflow.__version__}")
print(f"Optuna:   {optuna.__version__}")
print("CatBoost: OK")
print("InsuranceConformalPredictor: OK")
print("insurance-datasets: load_motor available")
print("All libraries ready.")
```

You should see version numbers printed for each library. If you see `ModuleNotFoundError`, the install cell did not complete successfully. Run it again and restart again.
