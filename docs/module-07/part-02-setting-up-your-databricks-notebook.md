## Part 2: Setting up your Databricks notebook

### Creating the notebook

If Databricks is open from Module 6, create a new notebook. In the left sidebar, click **Workspace**. Navigate to your user folder under `/Users/your.email@company.com/`. Click the **+** button and choose **Notebook**. Name it `module-07-rate-optimisation`. Leave the language as Python. Click **Create**.

The notebook opens with one empty cell. At the top of the notebook you will see a cluster selector. If it shows "Detached," click it and choose your cluster. Wait for the cluster name to show with a green circle. Do not run any cells until the cluster is connected.

If the cluster is not in the list, it may have auto-terminated. Go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Come back to the notebook once the cluster shows "Running."

### Installing the libraries

In the first cell, type this and run it by pressing **Shift+Enter**:

```python
%pip install rate-optimiser catboost polars scipy --quiet
dbutils.library.restartPython()
```

Wait for the installation output to complete, then for the restart message. This takes 60-90 seconds. After the restart, any variables from before are gone — that is expected.

For a local development environment instead of Databricks:

```bash
uv add rate-optimiser catboost polars scipy
```

### Confirming the imports work

In a new cell, paste this and run it:

```python
import numpy as np
import polars as pl
import scipy
from catboost import CatBoostClassifier
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
    EfficientFrontier,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print(f"SciPy:   {scipy.__version__}")
print("rate-optimiser: imported OK")
```

**What you should see:**

```
NumPy:   1.26.x
Polars:  0.20.x
SciPy:   1.x.x
rate-optimiser: imported OK
```

If you see `ModuleNotFoundError: No module named 'rate_optimiser'`, the install did not complete. Run the `%pip install` cell again and restart again.