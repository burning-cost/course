## Part 2: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder (or your Git repo if you set one up in Module 1).

Click the **+** button and choose **Notebook**. Name it `module-03-gbm-catboost`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Check the cluster selector at the top. If it says "Detached," click it and select your cluster from the dropdown. Once the cluster name appears in green, you are connected.

If the cluster is not running, go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Come back to the notebook once it shows "Running."

### Install the libraries

The libraries we need for this module are not all pre-installed on Databricks. In the first cell of your notebook, type this and run it (Shift+Enter):

```python
%pip install catboost optuna insurance-cv mlflow polars
```

You will see a long stream of output as pip downloads and installs each package. Wait for it to finish completely. At the end you will see:

```
Note: you may need to restart the Python kernel to use updated packages.
```

In the next cell, run:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables from before the restart are gone - this is expected. Always put your `%pip install` cell at the very top of the notebook, before any other code, so you only need to restart once per session.

Here is what each library does:

- **catboost** - the GBM library we use throughout this module. Built by Yandex, it handles categorical features natively and has a clean API for Poisson and Tweedie loss functions.
- **optuna** - a hyperparameter tuning framework. It uses Bayesian optimisation to search the parameter space efficiently rather than trying every combination in a grid.
- **insurance-cv** - a small library that generates walk-forward cross-validation splits specifically designed for insurance data, with an IBNR buffer year. We explain this fully in Part 6.
- **mlflow** - the experiment tracking library built into Databricks. It logs model parameters, metrics, and the model artefact itself so every run is reproducible and auditable.
- **polars** - the data manipulation library from Module 2. If it is already at cluster level, the install is a no-op.

### Confirm the imports work

In the next cell, type this and run it:

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import optuna
import json
from datetime import date
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from insurance_cv import WalkForwardCV
from mlflow.tracking import MlflowClient

print(f"Polars:   {pl.__version__}")
print(f"CatBoost: __version__ not exposed, but imported OK")
print(f"Optuna:   {optuna.__version__}")
print("All imports OK")
```

You should see version numbers with no errors. If you see `ModuleNotFoundError`, the install cell did not complete - check that you ran it before the restart, and if so run `%pip install` again.

**Why `import mlflow.catboost` is a separate line:** `mlflow` does not automatically load its CatBoost integration. The submodule `mlflow.catboost` must be imported explicitly before you can call `mlflow.catboost.log_model()`.