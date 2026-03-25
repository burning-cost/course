## Part 17: Summary

### What we built

We built a complete model monitoring pipeline for a deployed CatBoost frequency model. Starting from a trained model and two time windows of synthetic UK motor data, we computed four metrics:

- **PSI** on the predicted score distribution: has the risk distribution changed?
- **CSI** on each input feature: which features have driven any distribution shift?
- **A/E ratio** with confidence intervals: is the model calibrated correctly?
- **Gini drift** with a bootstrap z-test (Algorithm 2, arXiv 2510.04556): has the model's discriminatory power changed?

We assembled these into a `MonitoringReport` with automated traffic lights and a machine-readable JSON summary. We persisted the results to Delta tables with versioning and a 7-year retention policy. We scheduled the notebook as a Databricks job running on the 1st of each month at 06:00 UK time. We set up SQL Alerts for amber and red conditions. We built a recalibration trigger framework that logs factors to a Delta table that the pricing pipeline reads automatically. And we showed how to extract a complete regulatory evidence pack from the monitoring history.

### The decision framework in one place

| PSI | A/E | Gini | Interpretation | Action |
|-----|-----|------|----------------|--------|
| Low | OK | OK | Book is stable | Watch |
| Elevated | OK | OK | Mix shift, model handling it | Log CSI, update reference |
| Elevated | Elevated | OK | Mix shift causing calibration error | Recalibrate |
| Low | Elevated | Dropping | Concept drift | Retrain |
| Elevated | Elevated | Dropping | Severe drift | Retrain + escalate |

No monitoring framework catches everything. The framework we built is designed to be reliable on the signals it does catch and to avoid generating false alarms that cause alert fatigue. A pricing analyst who sees green every month for six months and then sees an amber should treat that amber seriously. That is the point.

### The code in sequence

Here is the full sequence of cells in the monitoring notebook, in order:

1. `%pip install insurance-monitoring insurance-datasets catboost polars mlflow`
2. `dbutils.library.restartPython()`
3. Imports
4. Configuration (CATALOG, SCHEMA, dates, model name)
5. Schema creation
6. Load data: `df = load_motor(polars=True)`, split into reference and current
7. Load model from MLflow registry
8. Generate predictions: `pred_ref`, `pred_cur`
9. PSI on score distribution
10. CSI for each feature
11. A/E ratio at portfolio level and by segment
12. Gini drift test
13. Build `MonitoringReport`: read `report.recommendation`, `report.results_`
14. Print and save the report as JSON (`report.to_dict()`, `report.to_polars()`)
15. Write summary to `monitoring_log` Delta table
16. Write CSI detail to `csi_results` Delta table
17. Write A/E detail to `ae_results` Delta table
18. Compute recalibration recommendation
19. Send alert if AMBER or RED

The notebook runs in sequence from top to bottom. Each cell depends on the variables set in previous cells. There are no circular dependencies.

### What comes next

The natural next step from this module is continuous integration: running the monitoring pipeline automatically after every data refresh, not just on a monthly schedule. This requires a data pipeline that triggers the monitoring job when new claims data is loaded - implemented using Databricks Delta Live Tables and job orchestration, which is covered in Module 12.

The other next step is model retraining governance: not just detecting that retraining is needed, but having a documented process for doing it, validating the new model against the old one, running a champion-challenger period, and retiring the old model from the registry. Module 13 covers the full model lifecycle from this angle.

### One final thought on what monitoring is for

The purpose of monitoring is not to produce reports. It is to maintain a pricing model that prices risks correctly, protects policyholders from being systematically overcharged or undercharged, and meets the regulator's expectations of a documented, proportionate, and responsive model governance framework.

The report is evidence that you are doing this. The actual work is looking at what the report says and making a judgement call about what to do next. The tools in this module give you the evidence. The judgement is yours.
