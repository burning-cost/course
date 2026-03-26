## Part 2: Setting up your Databricks notebook

### Creating the notebook

If Databricks is open from Module 6, create a new notebook. In the left sidebar, click **Workspace**. Navigate to your user folder under `/Users/your.email@company.com/`. Click the **+** button and choose **Notebook**. Name it `module-07-rate-optimisation`. Leave the language as Python. Click **Create**.

The notebook opens with one empty cell. At the top of the notebook you will see a cluster selector. If it shows "Detached," click it and choose your cluster. Wait for the cluster name to show with a green circle. Do not run any cells until the cluster is connected.

If the cluster is not in the list, it may have auto-terminated. Go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Come back to the notebook once the cluster shows "Running."

### Installing the libraries

In the first cell, type this and run it by pressing **Shift+Enter**:

```python
%pip install insurance-optimise polars --quiet
dbutils.library.restartPython()
```

Wait for the installation output to complete, then for the restart message. This takes 60-90 seconds. After the restart, any variables from before are gone — that is expected.

For a local development environment instead of Databricks:

```bash
uv add insurance-optimise polars
```

### Confirming the imports work

In a new cell, paste this and run it:

```python
import numpy as np
import polars as pl
import insurance_optimise
from insurance_optimise import (
    PortfolioOptimiser,
    ConstraintConfig,
    EfficientFrontier,
    ClaimsVarianceModel,
)

print(f"NumPy:             {np.__version__}")
print(f"Polars:            {pl.__version__}")
print(f"insurance-optimise: {insurance_optimise.__version__}")
print("All imports OK")
```

**What you should see:**

```bash
NumPy:              1.26.x
Polars:             0.20.x
insurance-optimise: 0.x.x
All imports OK
```

If you see `ModuleNotFoundError: No module named 'insurance_optimise'`, the install did not complete. Run the `%pip install` cell again and restart again.
