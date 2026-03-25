## Part 2: Setting up the notebook

### Creating the notebook

Open Databricks. In the left sidebar, click **Workspace**, navigate to your user folder (`/Users/your.email@company.com/`), click **+**, and choose **Notebook**. Name it `module-09-demand-elasticity`. Leave the default Python language. Click **Create**.

Attach your cluster. If it has auto-terminated after 2 hours idle, go to **Compute**, find the cluster, click **Start**, and wait 3–5 minutes.

### Installing the libraries

In the first cell:

```python
%pip install insurance-causal catboost econml polars --quiet
dbutils.library.restartPython()
```

`econml` is Microsoft Research's causal machine learning library. It provides the `CausalForestDML` estimator underneath `RenewalElasticityEstimator`. It has several dependencies — installation takes 90–120 seconds on Databricks Free Edition. Wait for the completion message and the Python restart confirmation before running anything else.

If `econml` fails with a timeout, rerun the cell. Databricks Free Edition occasionally has slow wheel resolution.

### Imports

```python
%md
## Module 9: Demand Elasticity — Imports
```

```python
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from insurance_causal.elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age
from insurance_causal.elasticity.fit import RenewalElasticityEstimator
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics, TreatmentVariationReport
from insurance_causal.elasticity.surface import ElasticitySurface
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve

print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print("All imports: OK")
```

You should see both version strings and "All imports: OK". If `from insurance_causal.elasticity.fit import RenewalElasticityEstimator` raises an `ImportError`, check that the `%pip install` cell completed without errors.

### A note on computation time

The `RenewalElasticityEstimator` with `cate_model="causal_forest"` fits a CausalForestDML model with CatBoost nuisance estimators using 5-fold cross-fitting. On 50,000 records on Databricks Free Edition, this takes 5–8 minutes. Start the fit cell and read ahead — Parts 4 and 5 are directly relevant to understanding the output.

The pre-flight diagnostic (Part 5) and the data generation (Part 4) each take under 30 seconds and should be run first.
