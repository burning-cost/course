## Part 15: Databricks deployment

### PyMC on Databricks — practical setup

PyMC 5.x runs on the standard Databricks ML runtime (DBR 14.x or later). Install it in the first cell of every notebook that uses it:

```python
%pip install pymc arviz --quiet
dbutils.library.restartPython()
```

The first import of PyMC in a session compiles PyTensor computation graphs. This takes 30-60 seconds on first run in a fresh cluster session. Subsequent cells run faster.

### Parallelising chains

On a multi-core cluster, NUTS chains run in parallel. To use all available cores:

```python
import multiprocessing
n_cores = multiprocessing.cpu_count()
print(f"Available cores: {n_cores}")

with hierarchical_model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=min(4, n_cores),
        cores=min(4, n_cores),   # run chains in parallel
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
    )
```

On a 4-core single-node cluster: 4 chains run in parallel, cutting wall-clock time by roughly 3-4×. On Databricks Free Edition (typically 1-2 cores), the chains run sequentially or with limited parallelism. For production models, use a standard cluster with 4-8 cores.

### MLflow tracking

Every hierarchical model fit should be tracked in MLflow. The convergence diagnostics are the most important artefacts — a model that failed convergence should not be usable downstream.

```python
import mlflow

mlflow.set_experiment("/pricing/credibility-bayesian/module06")

with mlflow.start_run(run_name="hierarchical_frequency_v1"):

    # Log convergence diagnostics as metrics
    mlflow.log_metric("max_rhat", max_rhat)
    mlflow.log_metric("min_ess_bulk", min_ess_bulk)
    mlflow.log_metric("n_divergences", n_div)
    mlflow.log_metric("n_districts", n_districts_model)
    mlflow.log_metric("sigma_district_mean",
                      float(trace.posterior["sigma_district"].mean()))
    mlflow.log_metric("grand_mean_rate",
                      float(np.exp(trace.posterior["alpha"].mean())))

    # Log the full posterior as an ArviZ netCDF artefact.
    # This lets you reload the posterior for any downstream analysis
    # without re-running MCMC.
    trace.to_netcdf("/tmp/posterior_module06.nc")
    mlflow.log_artifact("/tmp/posterior_module06.nc", "posteriors")

    # Log the results table
    results.write_csv("/tmp/credibility_results_module06.csv")
    mlflow.log_artifact("/tmp/credibility_results_module06.csv", "results")

    print(f"MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")
```

**What this does:** Logs the convergence diagnostics, the full posterior (as a netCDF file), and the results table to MLflow. The netCDF file means you can reload the posterior at any time without re-running MCMC — important because MCMC takes minutes but loading from disk takes seconds.

### Unity Catalog for credibility-weighted estimates

Credibility-weighted factor tables belong in Unity Catalog with the same governance as any other rating artefact:

```python
from datetime import date

RUN_DATE = str(date.today())
MODEL_NAME = "hierarchical_freq_module06_v1"

# Hard gate: do not write unconverged posteriors downstream.
# A model that has not converged produces estimates that are wrong in ways
# that are hard to detect. Fail loudly rather than propagate bad estimates.
if max_rhat > 1.01 or n_div > 0:
    raise ValueError(
        f"Convergence failure: max_rhat={max_rhat:.4f}, divergences={n_div}. "
        "Credibility estimates not written to Unity Catalog."
    )

results_out = results.with_columns([
    pl.lit(MODEL_NAME).alias("model_name"),
    pl.lit("hierarchical_poisson").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(max_rhat).alias("max_rhat"),
    pl.lit(n_div).alias("n_divergences"),
])

(
    spark.createDataFrame(results_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_credibility_estimates")
)

print(f"Written {results_out.height} rows to main.pricing.module06_credibility_estimates")
```

**What this does:** Writes the credibility-weighted estimates to a Delta table in Unity Catalog. The hard gate (`raise ValueError` if convergence fails) ensures a model with bad convergence cannot write downstream. This is not defensive programming — it is the only way to prevent silent errors from propagating through a pricing pipeline.