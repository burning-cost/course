## Part 2: Setting up the notebook

### Creating the notebook on Databricks

If you are coming from Module 8, your cluster should still be running. Open Databricks. In the left sidebar, click **Workspace**, navigate to your user folder (`/Users/your.email@company.com/`), click **+**, and choose **Notebook**. Name it `module-09-demand-elasticity`. Leave the default Python language. Click **Create**.

The notebook opens with one empty cell. At the top, confirm the cluster is attached and shows a green circle. If you see "Detached," click the cluster selector and choose your cluster. If your cluster has auto-terminated (they stop after 2 hours of inactivity by default), go to **Compute**, find your cluster, click **Start**, and wait 3-5 minutes.

### Installing the libraries

In the first cell, type the following and press **Shift+Enter** to run it:

```python
%pip install insurance-demand insurance-elasticity catboost econml polars --quiet
dbutils.library.restartPython()
```

This installs five packages. `econml` is Microsoft Research's causal ML library - it provides the `CausalForestDML` estimator that powers the heterogeneous elasticity estimation. It has several dependencies and takes about 90 seconds to install on Databricks Free Edition. Wait for the output to show a completion message, then wait for the Python restart confirmation.

After the restart, all previous variables are gone. That is expected. Do not run any earlier cells until you see "Python interpreter restarted" in the output.

If the install fails with a timeout, run the cell again. Databricks Free Edition occasionally has slow package resolution. If `econml` specifically fails, check that the wheel is available for your Python version with `!pip show econml`.

### Confirming imports work

In a new cell, paste and run:

```python
import numpy as np
import polars as pl

from insurance_demand import ConversionModel, RetentionModel, ElasticityEstimator
from insurance_demand import DemandCurve, OptimalPrice
from insurance_demand.datasets import generate_conversion_data, generate_retention_data
from insurance_demand.compliance import ENBPChecker

from insurance_elasticity.data import make_renewal_data
from insurance_elasticity.fit import RenewalElasticityEstimator
from insurance_elasticity.diagnostics import ElasticityDiagnostics
from insurance_elasticity.surface import ElasticitySurface
from insurance_elasticity.optimise import RenewalPricingOptimiser
from insurance_elasticity.demand import demand_curve, plot_demand_curve

print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print("All imports: OK")
```

You should see the version numbers and the "All imports: OK" message. If any import fails, check the pip install output for errors before proceeding.