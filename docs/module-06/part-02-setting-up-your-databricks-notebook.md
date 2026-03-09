## Part 2: Setting up your Databricks notebook

### Creating the notebook

If Databricks is already open from Module 5, you can create a new notebook. If you are starting fresh, go to your Databricks workspace URL and log in.

In the left sidebar, click the **Workspace** icon (it looks like a folder). Navigate to a folder you have write access to — usually your personal folder under `/Users/your.email@company.com/`. Right-click (or click the three dots next to the folder name) and select **Create > Notebook**. Name it `module-06-credibility-bayesian`. Leave the default language as Python.

The notebook opens in the editor. You will see an empty cell with a grey triangle (run button) on the left.

### Attaching a cluster

If your cluster from Module 5 is still running, click **Connect** in the top right and select it. If it has terminated (clusters auto-terminate after inactivity), start a new cluster: click **Connect > Create new cluster** (or go to the Compute section in the left sidebar). Use the default settings — a single-node cluster with the ML runtime (DBR 15.x) is sufficient for this module.

Databricks Free Edition (Community Edition) clusters have one driver node with no workers. That is fine for everything in this module. MCMC runs on the driver node.

### Installing PyMC

PyMC is not installed on the default Databricks ML runtime. Install it in the first cell of your notebook. Click in the first cell and type:

```python
%pip install pymc arviz --quiet
dbutils.library.restartPython()
```

**What this does:** `%pip install` installs packages into the running Python environment on the cluster. `--quiet` suppresses the progress output so the cell is less noisy. `dbutils.library.restartPython()` restarts the Python interpreter after installation — this is required because Python does not automatically make newly installed packages available in the current session.

**Run this cell** by pressing Shift+Enter or clicking the run button (triangle). This takes about 60-90 seconds. The cell will show the pip installation output and then a message saying the Python kernel has restarted.

**What you should see after it finishes:** The notebook will show a message like "Python interpreter restarted." All your cells after this one will work with PyMC available.

**Important:** After `dbutils.library.restartPython()` runs, the Python kernel resets. Any variables you defined in earlier cells are gone. This is why the installation cell must always be the first cell, and you must run the remaining cells after it completes.

### Importing libraries

Create a new cell (click the + icon below the first cell or press B to add a cell below). Paste in:

```python
import numpy as np
import polars as pl
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("NumPy:", np.__version__)
print("Polars:", pl.__version__)
print("PyMC:", pm.__version__)
print("ArviZ:", az.__version__)
```

**What this does:** Imports the four main libraries for this module. NumPy handles numerical arrays. Polars handles DataFrames (faster than pandas for the data manipulation we need). PyMC is the probabilistic programming library for Bayesian models. ArviZ provides MCMC diagnostics and plotting.

**Run this cell.** It takes about 30-60 seconds on first run because PyMC compiles PyTensor computation graphs on import.

**What you should see:**
```
NumPy: 1.26.x
Polars: 0.20.x
PyMC: 5.x.x
ArviZ: 0.18.x
```

The exact version numbers will differ depending on when you installed. What matters is that PyMC shows version 5.x — this module uses PyMC 5 syntax, which differs from PyMC 3 in a few places. If you see version 3.x, run `%pip install "pymc>=5" --quiet` and restart again.