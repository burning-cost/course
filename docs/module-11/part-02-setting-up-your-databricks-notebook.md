## Part 2: Setting up your Databricks notebook

### Creating the notebook

Open Databricks and navigate to your workspace folder under `/Users/your.email@company.com/`. Click **+** and choose **Notebook**. Name it `module-11-exposure-curves`. Leave the language as Python. Click **Create**.

At the top of the notebook, check the cluster selector. If it shows "Detached," click it and choose your cluster. If you need to start the cluster: go to **Compute** in the left sidebar, find your cluster, click **Start**, and wait 3-5 minutes. Do not run cells until the cluster shows a green circle.

### Installing the library

In the first cell, run:

```python
%pip install insurance-ilf insurance-datasets matplotlib --quiet
dbutils.library.restartPython()
```

This installs three packages:
- `insurance-ilf`: the exposure curve library we use throughout this module
- `insurance-datasets`: the synthetic UK commercial property dataset
- `matplotlib`: for exposure curve plots and Lee diagrams

Wait for installation to complete and for the restart message before continuing. The restart clears all variables -- that is expected.

For a local environment instead of Databricks:

```bash
uv add insurance-ilf insurance-datasets matplotlib
```

### Confirming the imports work

In a new cell:

```python
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from insurance_ilf import (
    MBBEFDDistribution,
    swiss_re_curve,
    empirical_exposure_curve,
    fit_mbbefd,
    fit_exposure_curve,
    ilf_table,
    excess_loss_factor,
    layer_expected_loss,
    per_risk_xl_rate,
    GoodnessOfFit,
    compare_curves,
    lee_diagram,
)
from insurance_ilf.curves import all_swiss_re_curves

print(f"NumPy:         {np.__version__}")
print(f"Polars:        {pl.__version__}")
print("insurance-ilf: imported OK")
```

**What you should see:**

```bash
NumPy:         1.26.x
Polars:        0.20.x
insurance-ilf: imported OK
```

If you see `ModuleNotFoundError: No module named 'insurance_ilf'`, the install did not complete. Re-run the `%pip install` cell and restart again.