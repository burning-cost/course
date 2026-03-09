## Part 4: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder.

Click the **+** button and choose **Notebook**. Name it `module-10-interactions`. Keep the default language as Python. Click **Create**.

Connect to your cluster. If it says "Detached" at the top right, click it and select your cluster. Wait for the green icon before continuing.

### Install the libraries

In the first cell, put both the install and the restart together:

```python
%pip install insurance-interactions glum polars numpy torch mlflow insurance-datasets --quiet
dbutils.library.restartPython()
```

Both commands must be in the same cell. If they are in separate cells, a learner may accidentally run the import cell before the restart completes. This clears the Python session — that is expected. Do not run any other cells until you see the restart confirmation.

The `insurance-interactions` library installs:
- **torch** — PyTorch, used to train the CANN. PyTorch is a large package; the first install may take 3-5 minutes
- **glum** — the GLM library for testing interactions
- **polars** — the data manipulation library we have used throughout this course
- **scipy** — for the chi-squared distribution in LR tests

`insurance-datasets` is the standard synthetic motor portfolio used throughout the course.

If you want SHAP interaction validation (Part 14), you also need:

```python
%pip install "insurance-interactions[shap]" catboost shapiq --quiet
```

We will note where the SHAP validation section begins and what to do if you have not installed these.

### Confirm the imports work

In a new cell:

```python
import polars as pl
import numpy as np
import mlflow
from insurance_interactions import (
    InteractionDetector,
    DetectorConfig,
    build_glm_with_interactions,
)
from insurance_datasets import load_motor

print("insurance-interactions version:", __import__("insurance_interactions").__version__)
print("insurance-datasets: load_motor available")
```

You should see:

```bash
insurance-interactions version: 0.1.0
insurance-datasets: load_motor available
```

If you get `ModuleNotFoundError`, the pip install did not complete before the restart. Re-run the `%pip install` cell and the restart cell, then try the import again.
