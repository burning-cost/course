## Part 15: Logging to MLflow

Log both models — the baseline GLM and the interaction-enhanced GLM — to MLflow so the comparison is auditable.

```python
import mlflow
import mlflow.sklearn

EXPERIMENT_NAME = "module_10_interactions"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="baseline_glm") as run_base:
    # Log baseline GLM metrics
    mlflow.log_metric("deviance", float(base_deviance))
    mlflow.log_metric("aic",      float(base_aic))
    mlflow.log_metric("n_params", int(len(glm_base.coef_) + 1))
    mlflow.log_param("family", "poisson")
    mlflow.log_param("interactions", "none")
    mlflow.sklearn.log_model(glm_base, "model")
    base_run_id = run_base.info.run_id
    print(f"Baseline run ID: {base_run_id}")

with mlflow.start_run(run_name="glm_with_interactions") as run_int:
    # Log enhanced GLM metrics
    int_deviance = comparison.filter(pl.col("model") == "glm_with_interactions")["deviance"][0]
    int_aic      = comparison.filter(pl.col("model") == "glm_with_interactions")["aic"][0]
    int_n_params = comparison.filter(pl.col("model") == "glm_with_interactions")["n_params"][0]

    mlflow.log_metric("deviance",               float(int_deviance))
    mlflow.log_metric("aic",                    float(int_aic))
    mlflow.log_metric("n_params",               int(int_n_params))
    mlflow.log_metric("delta_deviance",         float(int_deviance - base_deviance))  # negative = better
    mlflow.log_metric("delta_deviance_pct",     float(100 * (int_deviance - base_deviance) / base_deviance))
    mlflow.log_metric("n_interaction_pairs",    len(suggested))
    mlflow.log_param("family",               "poisson")
    mlflow.log_param("interactions",         str(suggested))
    mlflow.log_param("baseline_run_id",      base_run_id)

    # Log the interaction table as an artifact
    table_path = "/tmp/interaction_table.csv"
    table.to_pandas().to_csv(table_path, index=False)
    mlflow.log_artifact(table_path, "interaction_detection")

    mlflow.sklearn.log_model(enhanced_glm, "model")
    int_run_id = run_int.info.run_id
    print(f"Enhanced GLM run ID: {int_run_id}")
```

### View the comparison in the MLflow UI

In the Databricks left sidebar, click **Experiments**. Find `module_10_interactions`. You will see two runs. Click the checkboxes next to both and then click **Compare**. The comparison view shows deviance, AIC, and n_params side by side.

Under the enhanced GLM run, click **Artifacts**. Under `interaction_detection/`, you will find `interaction_table.csv` — the full ranked table with NID scores, LR test results, and recommendations. This is the audit trail for the interaction selection.