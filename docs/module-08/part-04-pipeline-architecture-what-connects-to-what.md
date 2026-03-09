## Part 4: Pipeline architecture -- what connects to what

Before writing any pipeline code, you need to understand the architecture as a whole. The following diagram shows how each stage feeds the next.

```
Stage 1: Configuration
    |
    v
Stage 2: Data ingestion --> raw_policies (Delta table, version logged)
    |
    v
Stage 3: Feature engineering --> features (Delta table, version logged)
    |
    +---> Stage 4: Walk-forward CV (validation metrics)
    |         |
    |         v
    +---> Stage 5: Optuna tuning (best_freq_params, best_sev_params)
    |         |
    |         v
    +---> Stage 6: Final models --> freq_run_id, sev_run_id (MLflow)
              |
              +---> Stage 7: SHAP relativities --> freq_predictions (Delta)
              |
              +---> Stage 8: Conformal intervals --> conformal_intervals (Delta)
              |
              +---> Stage 9: Rate optimisation --> rate_change_factors (Delta)
                        |
                        v
                   Stage 10: Audit record --> pipeline_audit (Delta)
```

Every stage reads from a well-defined input and writes to a well-defined output. Stage 3's output (features Delta table) feeds stages 4, 5, and 6. Stage 6's output (trained models, logged in MLflow) feeds stages 7, 8, and 9. The final audit record references every upstream output.

This structure means that if any stage fails, you can restart from that stage without rerunning everything before it. It also means that every output is traceable: given any Delta table row or MLflow model, you can follow the references back to the configuration that produced it.