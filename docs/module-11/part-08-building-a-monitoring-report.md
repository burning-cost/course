## Part 8: Building a MonitoringReport

### Assembling the metrics

We now have four sets of results sitting in variables:

- `psi_score` - PSI on the predicted score distribution
- `csi_scores` - CSI for each feature
- `ae_portfolio` - A/E ratio at portfolio level
- `gini_drift` - Gini change between reference and current

The `MonitoringReport` class takes all of these and produces a structured summary with automated traffic lights. The report is designed to be read by a pricing analyst who has not been staring at the notebook all month.

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    model_name=MODEL_NAME,
    reference_date=REFERENCE_DATE,
    current_date=CURRENT_DATE,
)

# Add the PSI result
report.add_psi(psi_score)

# Add all CSI results
for feature, result in csi_scores.items():
    report.add_csi(result)

# Add the A/E result
report.add_ae(ae_portfolio)

# Add the Gini drift result
report.add_gini_drift(gini_drift)
```

### Generating the summary

```python
summary = report.summary()

print("=" * 60)
print(f"MONITORING REPORT")
print(f"Model:     {summary['model_name']}")
print(f"Reference: {summary['reference_date']}")
print(f"Current:   {summary['current_date']}")
print(f"Run date:  {summary['run_date']}")
print("=" * 60)
print()

# Overall traffic light
overall = summary["overall_traffic_light"]
print(f"OVERALL STATUS: {overall}")
print()

# Individual metrics
print(f"{'Metric':<35} {'Value':>10}  {'Status'}")
print("-" * 60)

m = summary["metrics"]
print(f"{'Score PSI':<35} {m['psi_score']['value']:>10.4f}  {m['psi_score']['traffic_light']}")
print(f"{'A/E ratio':<35} {m['ae_ratio']['value']:>10.4f}  {m['ae_ratio']['traffic_light']}")
print(f"{'A/E CI lower':<35} {m['ae_ratio']['ci_lower']:>10.4f}")
print(f"{'A/E CI upper':<35} {m['ae_ratio']['ci_upper']:>10.4f}")
print(f"{'Gini (reference)':<35} {m['gini']['gini_ref']:>10.4f}")
print(f"{'Gini (current)':<35} {m['gini']['gini_cur']:>10.4f}  {m['gini']['traffic_light']}")
print(f"{'Gini p-value':<35} {m['gini']['p_value']:>10.4f}")
print()

# CSI summary
print("FEATURE CSI:")
print(f"{'Feature':<30} {'CSI':>8}  {'Status'}")
print("-" * 50)
for csi_item in sorted(summary["csi"], key=lambda x: x["csi"], reverse=True):
    print(f"{csi_item['feature']:<30} {csi_item['csi']:>8.4f}  {csi_item['traffic_light']}")
```

### How the overall traffic light is determined

The overall traffic light uses the following rules (in order of severity):

1. If any single metric is RED: overall is RED
2. If two or more metrics are AMBER: overall is AMBER
3. If one metric is AMBER and the rest are GREEN: overall is AMBER
4. If all metrics are GREEN: overall is GREEN

The thresholds for each metric:

| Metric | Green | Amber | Red |
|--------|-------|-------|-----|
| Score PSI | < 0.10 | 0.10 - 0.20 | > 0.20 |
| A/E ratio | CI contains 1.0 | CI excludes 1.0 but ratio in [0.90, 1.10] | CI excludes 1.0 and ratio outside [0.90, 1.10] |
| Gini drop | drop < 0.03 AND p > 0.10 | (p 0.05-0.10) OR (p < 0.05 and drop < 0.03) | p < 0.05 and drop >= 0.03 |
| Any CSI | < 0.10 | 0.10 - 0.20 | > 0.20 |

The A/E traffic light is based on the confidence interval, not just the point estimate. An A/E of 1.08 with a wide confidence interval (portfolio of 500 policies) is green; an A/E of 1.04 with a narrow confidence interval (portfolio of 50,000 policies) is amber.

### Saving the report as JSON

The summary dictionary is serialisable. Save it to a file and to Delta (we cover the Delta write in Part 10):

```python
import json

report_json = json.dumps(summary, indent=2, default=str)

# Save to DBFS for now; we persist to Delta in Part 10
report_path = f"/tmp/monitoring_report_{CURRENT_DATE}.json"
with open(report_path.replace("/tmp", "/dbfs/tmp"), "w") as f:
    f.write(report_json)

print(f"Report saved to {report_path}")
print()
print("Report JSON (truncated):")
print(report_json[:800])
```

### Generating a human-readable HTML report

The `MonitoringReport` class can also produce an HTML summary suitable for attaching to an email or pasting into a Confluence page. This is not a polished dashboard - it is a functional one-page status report:

```python
html = report.to_html()

# Save to DBFS
html_path = f"/dbfs/tmp/monitoring_report_{CURRENT_DATE}.html"
with open(html_path, "w") as f:
    f.write(html)

print(f"HTML report saved to {html_path}")

# Display in notebook
from IPython.display import HTML, display
display(HTML(html))
```

The HTML report has coloured cells for each metric (green, amber, red backgrounds), the full CSI table, and a timestamp. It is designed to be self-contained - no external CSS or JavaScript dependencies - so it renders identically whether you open it in a browser, paste it into a ticket, or attach it to an email.

Part 9 walks through how to read and act on the report. Part 10 shows how to write the results to Delta for trend analysis.
