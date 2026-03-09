## Part 0: Setting up your notebook

Before we write any modelling code, we need a notebook and the right libraries installed.

### Creating the notebook

In your Databricks workspace, go to the left sidebar and click **Workspace**. Navigate to your user folder (or your Git repo if you set one up in Module 1). Click the **+** button and choose **Notebook**.

Name it something like `module-02-glm-frequency-severity`. Keep the default language as Python. Click **Create**.

Check the cluster selector at the top of the notebook. If it says "Detached," click it and select your cluster. Once it shows the cluster name in green, you are connected and ready to run cells.

### Installing the libraries

The three libraries we need for this module are:

- **Polars** - for data manipulation. Think of it as Excel tables in Python. You define calculations on columns, filter rows, and group data - but it handles millions of rows instantly and the syntax is explicit about what is happening. We introduced Polars briefly in Module 1.
- **numpy** - for numerical computing. Arrays of numbers, mathematical functions (exp, log, etc.), and the random number generators we use to build our synthetic dataset. You will see it abbreviated as `np` throughout.
- **statsmodels** - the library that fits the GLMs. It contains Poisson, Gamma, Tweedie, and quasi-Poisson families, uses IRLS exactly as Emblem does, and produces coefficient tables, deviance statistics, and confidence intervals. You will see it abbreviated as `sm` or `smf`.

In a new cell at the top of your notebook, type this and run it (Shift+Enter):

```python
%pip install polars statsmodels scipy matplotlib
```

You will see a stream of output as pip downloads and installs the packages. Wait for it to finish. At the end it says something like:

```
Note: you may need to restart the Python kernel to use updated packages.
```

In the next cell, run:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables you defined before this point are gone - the session resets. This is expected. Always put your `%pip install` cell at the very top of the notebook so you only need to restart once per session.

**Why numpy is not in the install list:** numpy comes pre-installed with Databricks Runtime. You do not need to pip install it.

### Confirm the imports work

In the next cell, run this to confirm everything is installed correctly:

```python
import polars as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

print(f"Polars version: {pl.__version__}")
print(f"Statsmodels version: {sm.__version__}")
print("All imports OK")
```

You should see version numbers printed with no errors. If you see `ModuleNotFoundError`, the install cell did not work - check that you ran it before the restart, and if necessary run `%pip install` again.