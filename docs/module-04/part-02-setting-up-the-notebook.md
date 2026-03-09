## Part 2: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder (or the shared pricing folder your team uses).

Click the **+** button and choose **Notebook**. Name it `module-04-shap-relativities`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Check the cluster selector at the top right. If it says "Detached," click it and select your cluster from the dropdown. Wait for the cluster name to appear in green text before continuing.

If the cluster is not in the list, go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Return to the notebook once it shows "Running."

### Install the libraries

In the first cell, type this and run it (Shift+Enter):

```python
%pip install "shap-relativities[all]" catboost polars statsmodels mlflow insurance-datasets --quiet
```

You will see pip output for 30-60 seconds. Wait until you see:

```sql
Note: you may need to restart the Python kernel to use updated packages.
```

Once you see that, in the next cell type this and run it:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables from before the restart are cleared - that is expected. The `%pip install` must be the very first cell in your notebook, before any other code.

Here is what each library does:

- **shap-relativities** - the library we use throughout this module. The `[all]` extra pulls in CatBoost, the SHAP library itself, matplotlib, and statsmodels. It provides the `SHAPRelativities` class that handles the SHAP computation and relativity extraction.
- **catboost** - the GBM library from Module 3. We re-train the frequency model here, so this module is self-contained.
- **polars** - the data manipulation library from previous modules.
- **statsmodels** - for fitting the benchmark GLM. We compare GBM relativities to GLM relativities directly.
- **mlflow** - for logging the relativities table and validation results alongside the model.
- **insurance-datasets** - the standard synthetic UK motor portfolio used across all modules.

### Confirm the imports work

In a new cell, type this and run it (Shift+Enter):

```python
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import catboost as cb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shap_relativities import SHAPRelativities
from insurance_datasets import load_motor

print(f"Polars:          {pl.__version__}")
print(f"Statsmodels:     {sm.__version__}")
print(f"MLflow:          {mlflow.__version__}")
print("SHAPRelativities: imported OK")
print("insurance-datasets: load_motor available")
print("All imports OK")
```

You should see version numbers printed with no errors. If you see `ModuleNotFoundError: No module named 'shap_relativities'`, the install cell did not complete cleanly. Check that you ran the `%pip install` cell before the Python restart. If the error persists, run the install cell again and restart again.
