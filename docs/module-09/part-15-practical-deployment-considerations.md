## Part 15: Practical deployment considerations

The demand and elasticity models require ongoing monitoring and periodic re-fitting. This part covers what that looks like in practice.

### How often to refit

The conversion model should be refitted at least quarterly. Conversion rates shift with market conditions, competitor behaviour, PCW algorithm changes, and seasonal patterns. A model trained on data from 18 months ago may have learned a relationship between price ratio and conversion that no longer holds.

A practical refit schedule for a medium-sized motor book:

- **Conversion model**: quarterly refit on the most recent 12 months of quotes
- **Retention model**: semi-annual refit on the most recent 18 months of renewals
- **Elasticity model**: annual refit, triggered by major pricing events (bulk re-rate, PCW relationship change, post-FCA review)

The elasticity model is refitted less frequently because it requires sufficient residual treatment variation, which accumulates slowly from rate reviews and any A/B tests.

### What you need to store at quote time

The single most common data quality problem in demand modelling is not having the technical premium stored at quote time. The technical premium must be the value from the underwriting system at the moment the quote was issued. Retrospectively recalculated technical premiums - even if they use the same model - introduce errors because the model may have changed, the reference data may have changed, and the quote date effects are lost.

If your quotes table does not contain a `technical_premium_at_quote` column, the DML identification argument is weaker. You are stuck using log(quoted_price) as the treatment rather than log(quoted_price / technical_premium), and the confounding problem is harder to solve.

The fix is simple but requires buy-in from the technology team: add a column to the quote event table that captures the risk model output at that moment. It is a one-time data plumbing change that permanently improves your causal identification.

### MLflow experiment tracking

Log all model outputs to MLflow for the audit trail. Add a markdown cell:

```python
%md
## Part 15: MLflow logging
```

```python
import mlflow
import mlflow.sklearn

# Create an experiment if it does not already exist
mlflow.set_experiment("/Users/your.email@company.com/module-09-demand-elasticity")

with mlflow.start_run(run_name="conversion_elasticity_dml"):
    # Log the elasticity estimate
    mlflow.log_metric("elasticity_ate", est_conversion.elasticity_)
    mlflow.log_metric("elasticity_se", est_conversion.elasticity_se_)
    mlflow.log_metric("elasticity_ci_lower", est_conversion.elasticity_ci_[0])
    mlflow.log_metric("elasticity_ci_upper", est_conversion.elasticity_ci_[1])

    # Log parameters
    mlflow.log_param("n_folds", est_conversion.n_folds)
    mlflow.log_param("outcome_model", est_conversion.outcome_model)
    mlflow.log_param("treatment_col", est_conversion.treatment_col)
    mlflow.log_param("n_training_records", len(df_quotes))

    # Log the audit summary
    mlflow.log_metric("enbp_breach_count", n_breaches)
    mlflow.log_metric("portfolio_renewal_rate_predicted",
                      float(priced_df["predicted_renewal_prob"].mean()))

    print("Logged to MLflow.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

In a governance context, every elasticity estimate used in a pricing decision should have:
- The run ID from MLflow (so you can trace which model produced which output)
- The date of fitting and the date range of training data
- The ENBP audit summary signed off by the pricing actuary
- The treatment variation diagnostic output (confirming sufficient price variation)

### Writing results to Unity Catalog

```python
# Write the priced renewal portfolio to a Delta table
priced_df_pd = priced_df.to_pandas()

# spark.createDataFrame(priced_df_pd).write.format("delta") \
#     .mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .saveAsTable("pricing.motor.renewal_optimal_prices")

# Write the demand curve
demand_df_pd = demand_df.to_pandas()

# spark.createDataFrame(demand_df_pd).write.format("delta") \
#     .mode("overwrite") \
#     .saveAsTable("pricing.motor.renewal_demand_curve")

print("Tables to write:")
print("  pricing.motor.renewal_optimal_prices  -", len(priced_df), "rows")
print("  pricing.motor.renewal_demand_curve    -", len(demand_df), "rows")
```

The `spark.createDataFrame()` calls are commented out because the synthetic data is not in a real Unity Catalog. In your production environment, uncomment them and replace the table names with your actual schema.