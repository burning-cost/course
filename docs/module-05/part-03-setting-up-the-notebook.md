## Part 3: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder - this is usually listed under **Workspace > Users > your-email@company.com**.

Click the **+** button (at the top of the sidebar or next to your folder name) and choose **Notebook**. Name it `module-05-conformal-intervals`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. At the top of the notebook you will see a cluster selector - it usually shows "Detached" or your cluster name. If it says "Detached," click it and choose your cluster from the dropdown. Wait for the cluster name to appear with a green circle next to it. Do not run any cells until the cluster is connected.

If your cluster is not in the list, it may not be running. Go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes to start. Once the cluster shows "Running" with a green icon, come back to the notebook and connect to it.

### Install the libraries

In the first cell of your notebook, type this and run it by pressing **Shift+Enter**:

```python
%pip install "insurance-conformal[catboost]" catboost polars mlflow --quiet
```

You will see pip installation output scrolling for 30-60 seconds. Wait for it to finish completely. The last few lines will say something like:

```
Successfully installed insurance-conformal-0.x.x catboost-1.x.x ...
Note: you may need to restart the Python kernel to use updated packages.
```

Once you see that, run this in the next cell:

```python
dbutils.library.restartPython()
```

This restarts the Python session so the newly installed packages are available. Any variables from before the restart are gone - that is expected. The `%pip install` cell must always be the very first cell in the notebook, before any other code.

**What you should see after the restart:** the cell runs silently and the notebook is ready for the next cell. There is no output from `dbutils.library.restartPython()`.

### What each library does

- **insurance-conformal** - the core library for this module, from `github.com/burning-cost/insurance-conformal`. It provides the `InsuranceConformalPredictor` class, which calibrates conformal predictors, generates prediction intervals, and runs coverage diagnostics. The `[catboost]` extra installs the CatBoost integration.
- **catboost** - the gradient boosted tree library from Modules 3 and 4. We use it here for the base Tweedie model.
- **polars** - the data manipulation library from all previous modules.
- **mlflow** - the experiment tracking library built into Databricks.

### Confirm the imports work

In a new cell, type this and run it (Shift+Enter):

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import date
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor

print(f"Polars:   {pl.__version__}")
print(f"MLflow:   {mlflow.__version__}")
print(f"NumPy:    {np.__version__}")
print("InsuranceConformalPredictor: imported OK")
print("All imports OK")
```

**What you should see:**

```
Polars:   0.x.x
MLflow:   2.x.x
NumPy:    1.x.x
InsuranceConformalPredictor: imported OK
All imports OK
```

If you see `ModuleNotFoundError: No module named 'insurance_conformal'`, the install cell did not complete. Check that you ran the `%pip install` cell first, then `dbutils.library.restartPython()`. If the error persists, run the install cell again and restart again.