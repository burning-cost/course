## Part 3: Setting up your Databricks notebook

### Creating the notebook

If Databricks is open from Module 11, create a new notebook. In the left sidebar, click **Workspace**. Navigate to your user folder under `/Users/your.email@company.com/`. Click the **+** button and choose **Notebook**. Name it `module-12-spatial-territory`. Leave the language as Python. Click **Create**.

At the top of the notebook you will see a cluster selector. If it shows "Detached", click it and choose your cluster. Wait for the cluster name to appear with a green circle before running any cells.

If the cluster has auto-terminated, go to **Compute** in the left sidebar, find your cluster, and click **Start**. This takes 3--5 minutes.

### Installing the libraries

In the first cell:

```python
%pip install insurance-spatial pymc arviz matplotlib --quiet
dbutils.library.restartPython()
```

Wait for the restart message. The restart takes 30--60 seconds. After the restart, variables from before are gone -- that is expected. PyMC and ArviZ bring in several dependencies (pytensor, numpy, scipy) which are typically already installed in Databricks runtime environments but need to be explicitly requested here for version compatibility.

**Compatibility note:** If PyMC installs but ArviZ fails to import with a numpy compatibility error, add `numpy<2.0` to the install command and restart: `%pip install insurance-spatial pymc arviz matplotlib "numpy<2.0" --quiet`. This is a known issue in the PyMC 5 / ArviZ 0.18 ecosystem on environments with newer numpy.

**Compatibility note:** If PyMC installs but ArviZ fails to import with a numpy compatibility error, add `numpy<2.0` to the install command and restart: `%pip install insurance-spatial pymc arviz matplotlib "numpy<2.0" --quiet`. This is a known issue in the PyMC 5 / ArviZ 0.18 ecosystem on environments with newer numpy.

For a local environment instead:

```bash
uv add "insurance-spatial[geo]" pymc arviz matplotlib
```

The `[geo]` extra installs geopandas and libpysal, which are required for the real-boundary section in Part 10. You do not need them for the synthetic data sections.

### Confirming imports

In a new cell:

```python
import numpy as np
import polars as pl
from insurance_spatial import build_grid_adjacency, BYM2Model
from insurance_spatial.diagnostics import moran_i, convergence_summary
from insurance_spatial.relativities import extract_relativities
from insurance_spatial.plots import plot_relativities, plot_trace
import matplotlib.pyplot as plt
import arviz as az

print(f"NumPy:            {np.__version__}")
print(f"Polars:           {pl.__version__}")
print(f"insurance-spatial: imported OK")
print(f"ArviZ:            {az.__version__}")
```

**What you should see:**

```
NumPy:            1.26.x
Polars:           0.20.x
insurance-spatial: imported OK
ArviZ:            0.18.x
```

If you see `ModuleNotFoundError: No module named 'pymc'`, the `%pip install` cell did not complete successfully. Run it again (making sure the cluster is connected) and restart again.