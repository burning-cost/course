## Part 6: Compute basics

### What a cluster actually is

When you run code in a Databricks notebook, it does not run on your laptop. It runs on a **cluster** - a virtual computer (or collection of computers) in the cloud.

The cluster has:
- A specific amount of RAM and CPU cores
- A specific version of Databricks Runtime (which determines the Python version, pre-installed libraries, and so on)
- A start-up time (3-5 minutes, because a virtual machine is being provisioned)
- A cost (on paid tiers - charged by the hour)

When nothing is running and the cluster sits idle, it wastes money. For this reason, clusters are configured to **auto-terminate** after a period of inactivity - typically 30-60 minutes. When a cluster auto-terminates, your notebooks still exist and your Delta tables still exist, but you will need to restart the cluster (or attach to a different running one) before you can run code again.

### Free Edition compute

On Free Edition you have one cluster with fixed specifications. You cannot change the size. It auto-terminates after 2 hours of inactivity. When it terminates, you can restart it from the Compute panel - click the cluster name, then **Start**.

The Free Edition cluster is more than sufficient for learning: loading data, running models on small datasets, exploring the Databricks interface. It is not fast enough for training CatBoost on 500,000 policies - that kind of work requires a paid workspace.

### Starting and stopping a cluster

**To start:** Click **Compute** in the left sidebar. If the cluster shows status "Terminated", click the cluster name and then **Start**. Wait 3-5 minutes for it to reach "Running" status.

**To stop:** Click **Compute**, click the cluster name, click **Terminate** (or the stop button). On Free Edition this is rarely necessary since it auto-terminates anyway, but on paid tiers you should stop clusters when you are not using them.

**To check if a cluster is running from a notebook:** Look at the cluster selector at the top of the notebook. If it shows the cluster name in green, it is running and attached. If it shows "Detached" or the name in grey, either the cluster is stopped or the notebook is not connected to it.

### What happens when a cluster auto-terminates

If a cluster terminates while you have a notebook open:
- The notebook code and outputs that have already run are preserved
- Variables and in-memory data are lost (you will need to rerun the cells)
- Delta tables on disk are preserved
- The next time you run a cell, Databricks will prompt you to restart the cluster (this takes 3-5 minutes)

This is one reason to save important results to Delta tables rather than keeping them as in-memory DataFrames. A Delta table persists across cluster restarts. A Python variable does not.

### Installing libraries: per-notebook vs cluster-level

There are two ways to install libraries on Databricks:

**Per-notebook install using `%pip`** - this is what we used in Part 3. The library is installed for the current session. When the cluster restarts, you will need to run the `%pip install` cell again. This is fine for learning and for notebooks that are not part of automated pipelines.

```python
%pip install polars catboost matplotlib
```

**Cluster-level install** - the library is installed on the cluster itself, available to all notebooks, and persists across cluster restarts. On Free Edition, go to **Compute**, click the cluster, go to the **Libraries** tab, click **Install new**, select PyPI, and enter the package name.

For a team environment, cluster-level installs are better - everyone gets the same library version without each notebook needing its own install step. The Databricks Runtime ML versions (which we recommended in Part 2) pre-install many common libraries including scikit-learn, MLflow, and SHAP.

For this course, `%pip install` at the top of each notebook is fine.