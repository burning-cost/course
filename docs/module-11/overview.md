# Module 11: Model Monitoring and Drift Detection

## What this module covers

You have a trained CatBoost frequency model from Module 8. It is deployed, scoring new business, feeding the pricing pipeline. It produces predictions every day.

Now what?

This module covers the period after deployment: how you know the model is still working, how you detect when it stops working, and what you do about it. This is called model monitoring.

A model trained on 2022-2023 data encodes the relationships that existed in that world. That world changes. Drivers age, claim patterns shift, the mix of business changes, the Ogden rate moves. The model does not automatically update itself. If you do not monitor it, you will not know it has gone stale until loss ratios tell you - and by then you have been underpricing for months.

We use the `insurance-monitoring` library throughout. It provides PSI, CSI, A/E ratios with Poisson confidence intervals, Gini drift testing, and a combined `MonitoringReport` class that runs everything in one call.
[Download the notebook for this module](notebook.py)

## Structure

| Part | Topic |
|------|-------|
| 1 | Why monitoring matters |
| 2 | Setting up the notebook |
| 3 | What is model drift? |
| 4 | Population Stability Index (PSI) |
| 5 | Characteristic Stability Index (CSI) |
| 6 | Actual vs Expected ratios |
| 7 | Gini drift detection |
| 8 | Building a MonitoringReport |
| 9 | Interpreting the report |
| 10 | Writing results to Delta tables |
| 11 | Scheduling monitoring as a Databricks job |
| 12 | Setting up alerts |
| 13 | Recalibration triggers |
| 14 | Connecting to the pricing pipeline |
| 15 | Regulatory reporting |
| 16 | What we have not covered |
| 17 | Summary |

## Prerequisites

- Module 8: End-to-End Pipeline (the model we are monitoring lives there)
- Module 3: CatBoost (you need to understand what the model does)
- Module 2: Polars basics

## Libraries used

- `insurance-monitoring` - monitoring metrics and reporting
- `insurance-datasets` - synthetic UK motor data
- `catboost` - the model itself
- `polars` - data manipulation
- `mlflow` - metric logging
- `delta` - persisting results (Databricks built-in)
