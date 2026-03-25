## Part 8: Building a MonitoringReport

### The combined report

`MonitoringReport` is a dataclass that runs all monitoring checks at construction time. Pass your arrays and DataFrames in; it immediately computes A/E, Gini drift, CSI, and — optionally — a Murphy decomposition. There is no `.add_*()` pattern or `.summary()` method; results are available immediately via `results_`, `recommendation`, `to_dict()`, and `to_polars()`.

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=actual_ref,
    reference_predicted=pred_ref,
    current_actual=actual_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=df_reference,
    feature_df_current=df_current,
    features=FEATURE_NAMES,
    murphy_distribution="poisson",   # Murphy MCB/DSC decomposition — always available
    gini_bootstrap=False,            # True = add percentile CIs on Gini (slower)
)
```

The `feature_df_reference` and `feature_df_current` parameters are Polars DataFrames. The `features` list specifies which columns to include in CSI. If you omit these, CSI is skipped.

### Accessing the results

```python
# The decision recommendation
print("Recommendation:", report.recommendation)

# results_ is a nested dict with keys: ae_ratio, gini, csi, max_csi, (murphy)
results = report.results_

ae = results["ae_ratio"]
print(f"A/E: {ae['value']:.4f}  (CI: [{ae['lower_ci']:.4f}, {ae['upper_ci']:.4f}])  [{ae['band']}]")

gini = results["gini"]
print(f"Gini: {gini['current']:.4f}  (ref: {gini['reference']:.4f})  [{gini['band']}]")
print(f"Gini p-value: {gini['p_value']:.4f}")

if "max_csi" in results:
    mc = results["max_csi"]
    print(f"Max CSI: {mc['value']:.4f}  (worst feature: {mc['worst_feature']})  [{mc['band']}]")
```

### The recommendation

The `recommendation` property implements the three-stage decision tree from arXiv 2510.04556:

| Signal combination | Recommendation |
|---|---|
| All green | `NO_ACTION` |
| A/E drifted, Gini stable | `RECALIBRATE` |
| Gini degraded | `REFIT` |
| Multiple conflicting signals | `INVESTIGATE` |
| Amber signals, no red | `MONITOR_CLOSELY` |

When `murphy_distribution` is set, the Murphy decomposition sharpens the `RECALIBRATE` vs `REFIT` distinction: if global miscalibration dominates (GMCB > LMCB), recommend `RECALIBRATE` even if the Gini z-test has not crossed amber yet. If discrimination has fallen (DSC low), recommend `REFIT`.

### Flat Polars output

For writing to Delta or MLflow, use `to_polars()`:

```python
flat = report.to_polars()
print(flat)
# shape: (N, 3)
# ┌──────────────────────┬─────────┬──────────┐
# │ metric               ┆ value   ┆ band     │
# │ ---                  ┆ ---     ┆ ---      │
# │ str                  ┆ f64     ┆ str      │
# ├──────────────────────┼─────────┼──────────┤
# │ ae_ratio             ┆ 1.0832  ┆ amber    │
# │ ae_ratio_lower_ci    ┆ 1.0612  ┆ amber    │
# │ ae_ratio_upper_ci    ┆ 1.1056  ┆ amber    │
# │ gini_current         ┆ 0.3801  ┆ amber    │
# │ gini_reference       ┆ 0.4123  ┆ green    │
# │ ...                  ┆ ...     ┆ ...      │
```

### Full dict for JSON and Delta

```python
import json

report_dict = report.to_dict()
report_json = json.dumps(report_dict, indent=2, default=str)

report_path = f"/tmp/monitoring_report_{CURRENT_DATE}.json"
with open(report_path, "w") as f:
    f.write(report_json)

print(f"Report saved to {report_path}")
```

`to_dict()` returns a nested dict with keys `results`, `recommendation`, and `murphy_available`. It is designed for JSON serialisation and for logging to MLflow as run metrics.

### How the traffic lights work

The `results_` dict uses `band` keys throughout: `green`, `amber`, or `red`. Thresholds come from `MonitoringThresholds` (configurable, defaults to industry-standard settings):

| Metric | Green | Amber | Red |
|--------|-------|-------|-----|
| A/E (point estimate) | [0.95, 1.05] | [0.90, 1.10] | outside [0.80, 1.20] |
| Gini drift (p-value) | p >= 0.32 | 0.10 <= p < 0.32 | p < 0.10 |
| CSI (per feature) | < 0.10 | 0.10–0.25 | > 0.25 |
| PSI (score) | < 0.10 | 0.10–0.25 | > 0.25 |

The A/E classification uses the point estimate directly via `AERatioThresholds`. The default thresholds are 0.95/1.05 for green, 0.90/1.10 for amber, and 0.80/1.20 for red. Gini drift thresholds default to alpha=0.32 for amber and alpha=0.10 for red, following the one-sigma monitoring recommendation from arXiv 2510.04556.

To tighten PSI thresholds for a large book, or to use conventional statistical thresholds for a regulatory submission:

```python
from insurance_monitoring.thresholds import MonitoringThresholds, PSIThresholds, GiniDriftThresholds

# Tighter PSI for a large motor book with monthly monitoring
thresholds = MonitoringThresholds(
    psi=PSIThresholds(green_max=0.05, amber_max=0.15),
)

# Conventional statistical thresholds for a governance submission
thresholds_formal = MonitoringThresholds(
    gini_drift=GiniDriftThresholds(amber_p_value=0.10, red_p_value=0.05),
)

report = MonitoringReport(
    ...,
    thresholds=thresholds,
)
```

Part 9 walks through how to read and act on the report. Part 10 shows how to write the results to Delta for trend analysis.
