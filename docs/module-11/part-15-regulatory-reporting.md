## Part 15: Regulatory reporting

### What regulators want to see

SS1/23 does not specify a particular report format. What it requires is evidence that monitoring is happening, that thresholds are defined, and that breaches trigger a documented response. In a PRA review or an internal audit, you will need to produce:

1. **Monitoring framework documentation** - what metrics are computed, at what frequency, with what thresholds
2. **Monitoring run history** - evidence that monitoring was actually run on the agreed schedule
3. **Traffic light history** - the sequence of GREEN/AMBER/RED outcomes
4. **Breach response records** - what happened when an AMBER or RED was triggered
5. **Model action records** - recalibrations and retrainings, with rationale

The Delta tables from Parts 10 and 13 contain all of this information. This part shows how to extract it into a coherent evidence pack.

### FCA obligations for personal lines pricing teams

For a personal lines pricing team, the FCA's regulatory requirements are at least as directly relevant as PRA guidance — and in many firms will be the primary obligation.

The **general insurance pricing practices rules** (PS21/5, effective January 2022) prohibit price walking and require firms to evidence that renewal pricing is fair relative to equivalent new business pricing. A monitoring framework that tracks A/E ratios by policy tenure and renewal count is directly useful for demonstrating compliance.

The **Consumer Duty** (PS22/9, in force July 2023) goes further. It requires firms to demonstrate that their products provide fair value to customers — not just that pricing processes were followed, but that outcomes are fair. For a pricing model, this means being able to show that the model is not systematically overcharging identifiable cohorts of customers. Monitoring A/E ratios by segment (age band, region, vehicle group) is one of the most direct ways to produce this evidence: if the model is consistently predicting more claims than occur for a particular segment, those customers are being overcharged relative to their actual risk. The Consumer Duty requires firms to act on this, not just observe it.

In practice, your compliance team will want to see the monitoring framework documented in a way that maps explicitly to fair value assessments. The evidence pack generated below satisfies both PRA governance requirements and the FCA's expectation that pricing outcomes are monitored and acted on.

### Generating the monitoring framework documentation

This is a one-time document (updated when the framework changes) that describes the monitoring setup. Write it as a notebook cell that produces a formatted summary:

```python
framework_doc = f"""
MOTOR FREQUENCY MODEL - MONITORING FRAMEWORK
============================================

Model:            {MODEL_NAME}
Framework version: 1.0
Effective from:   {REFERENCE_DATE}
Last reviewed:    {RUN_DATE}

MONITORING SCHEDULE
-------------------
Frequency:    Monthly (automated Databricks job)
Run timing:   1st of month at 06:00 UK time
Job name:     motor-model-monitoring-monthly
Notebook:     module-11-model-monitoring

METRICS AND THRESHOLDS
----------------------
1. Score PSI (Population Stability Index on predicted frequencies)
   Green:  PSI < 0.10
   Amber:  0.10 <= PSI < 0.20
   Red:    PSI >= 0.20

2. A/E Ratio (Actual vs Expected claim frequency)
   Green:  95% CI contains 1.0
   Amber:  95% CI excludes 1.0 but ratio in [0.90, 1.10]
   Red:    Ratio outside [0.90, 1.10]

3. Gini Drift (DeLong test for change in AUC)
   Green:  p-value > 0.10, or Gini drop < 0.03
   Amber:  p-value 0.05-0.10, or p < 0.05 and drop < 0.03
   Red:    p-value < 0.05 and Gini drop >= 0.03

4. Feature CSI (Characteristic Stability Index per feature)
   Green:  CSI < 0.10
   Amber:  0.10 <= CSI < 0.20
   Red:    CSI >= 0.20

OVERALL STATUS
--------------
RED:    Any single metric is RED
AMBER:  Two or more AMBER metrics, or one AMBER metric (no RED)
GREEN:  All metrics GREEN

ACTION TRIGGERS
---------------
Recalibrate: A/E CI excludes 1.0 for two consecutive months, OR
             A/E point estimate outside [0.90, 1.10]
Retrain:     Statistically significant Gini drop for two consecutive months, OR
             A/E outside [0.85, 1.15], OR
             Two recalibrations without resolving A/E signal
Escalate:    Overall RED in any month (immediate), OR
             Three consecutive AMBER months (within 5 working days)

DATA
----
Reference window: Rolling 12 months prior to current period
Current window:   Previous calendar month
Minimum policies: 1,000 (monitoring output flagged if below this)
Minimum claims:   50 (Gini and A/E reliability warning if below this)

RESPONSIBILITY
--------------
Owner:          Head of Pricing
Monthly review: Pricing analyst (named in job configuration)
Escalation:     CRO / Actuarial Function Holder
"""

print(framework_doc)

# Save to DBFS for audit purposes
with open(f"/dbfs/tmp/monitoring_framework_v1.txt", "w") as f:
    f.write(framework_doc)
```

### Generating the annual evidence pack

At the end of each year, generate a summary of all monitoring runs. This is the document you hand to an auditor or produce for a PRA review:

```python
# Annual evidence pack query
year = "2024"

annual_query = f"""
SELECT
    current_date,
    overall_traffic_light,
    ae_ratio,
    ae_ci_lower,
    ae_ci_upper,
    gini_ref,
    gini_cur,
    gini_p_value,
    psi_score,
    reference_n,
    current_n,
    actual_claims,
    expected_claims
FROM {TABLES["monitoring_log"]}
WHERE model_name = '{MODEL_NAME}'
  AND YEAR(current_date) = {year}
ORDER BY current_date
"""

annual_df = spark.sql(annual_query)
annual_pl = pl.from_pandas(annual_df.toPandas())

print(f"ANNUAL MONITORING SUMMARY - {year}")
print(f"Model: {MODEL_NAME}")
print(f"Runs: {annual_pl.shape[0]}")
print()
print(f"{'Month':<12} {'Overall':<10} {'A/E':>8}  {'Gini cur':>10}  {'PSI':>8}")
print("-" * 55)

for row in annual_pl.iter_rows(named=True):
    print(f"{str(row['current_date'])[:10]:<12} "
          f"{row['overall_traffic_light']:<10} "
          f"{row['ae_ratio']:>8.4f}  "
          f"{row['gini_cur']:>10.4f}  "
          f"{row['psi_score']:>8.4f}")

# Count by traffic light
tl_counts = annual_pl.group_by("overall_traffic_light").len()
print()
print("Traffic light distribution:")
for row in tl_counts.iter_rows(named=True):
    print(f"  {row['overall_traffic_light']}: {row['len']} months")
```

### Breach response log

Any month that triggered an action (recalibration or retraining) needs a documented breach response. The recalibration history table (Part 13) provides part of this. Add a free-text reason field for the complete record:

```python
# Query breach history
breach_query = f"""
SELECT
    m.current_date             AS monitoring_date,
    m.overall_traffic_light,
    m.ae_ratio,
    r.recalibration_factor,
    r.reason                   AS action_reason,
    r.applied_by
FROM {TABLES["monitoring_log"]} m
LEFT JOIN {CATALOG}.{SCHEMA}.recalibration_history r
    ON m.model_name = r.model_name
    AND m.current_date = r.effective_from
WHERE m.model_name = '{MODEL_NAME}'
  AND (m.overall_traffic_light IN ('AMBER', 'RED') OR r.recalibration_factor IS NOT NULL)
ORDER BY m.current_date
"""

breach_df = spark.sql(breach_query)
breach_df.show(truncate=False)
```

### Writing to a governed Delta table for audit

The monitoring framework document and evidence pack should themselves be stored in Delta, not just in `/dbfs/tmp/`. Add a documentation table:

```python
doc_table = f"{CATALOG}.{SCHEMA}.monitoring_framework_versions"

doc_record = {
    "version":       "1.0",
    "effective_from": REFERENCE_DATE,
    "created_date":   RUN_DATE,
    "model_name":     MODEL_NAME,
    "content":        framework_doc,
    "author":         "pricing_team",
}

doc_df = spark.createDataFrame([Row(**doc_record)])

(
    doc_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(doc_table)
)

print(f"Framework documentation written to {doc_table}")
```

With this in place, a PRA reviewer or internal auditor can query the full audit trail in a single SQL session:

```sql
-- What did monitoring show in Q1 2024?
SELECT * FROM main.motor_monitoring.monitoring_log
WHERE YEAR(current_date) = 2024 AND MONTH(current_date) BETWEEN 1 AND 3;

-- Were any recalibrations applied in 2024?
SELECT * FROM main.motor_monitoring.recalibration_history
WHERE YEAR(effective_from) = 2024;

-- What version of the monitoring framework was in use?
SELECT * FROM main.motor_monitoring.monitoring_framework_versions
WHERE effective_from <= '2024-03-31'
ORDER BY effective_from DESC
LIMIT 1;
```

Three queries. Complete audit trail. This is what "documented model monitoring framework" looks like in practice.
