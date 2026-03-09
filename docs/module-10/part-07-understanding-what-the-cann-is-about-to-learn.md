## Part 7: Understanding what the CANN is about to learn

Before training, we can look at where the GLM is going wrong. This is the signal the CANN will pick up.

### Residual analysis by age band and vehicle group

```python
import matplotlib.pyplot as plt

df_diag = pl.DataFrame({
    "age_band":      X["age_band"].to_list(),
    "vehicle_group": X["vehicle_group"].to_list(),
    "y":             y,
    "mu_glm":        mu_glm,
    "exposure":      exposure_arr,
})

# Actual/expected by age band and vehicle group
ae_table = (
    df_diag
    .group_by(["age_band", "vehicle_group"])
    .agg([
        pl.sum("y").alias("observed"),
        pl.sum("mu_glm").alias("predicted"),
        pl.sum("exposure").alias("exposure"),
    ])
    .with_columns(
        (pl.col("observed") / pl.col("predicted")).alias("ae_ratio")
    )
    .sort(["age_band", "vehicle_group"])
)

# Show the worst cells
print("Cells with highest A/E ratio (GLM underpredicting):")
print(ae_table.sort("ae_ratio", descending=True).head(10))

print("\nCells with lowest A/E ratio (GLM overpredicting):")
print(ae_table.sort("ae_ratio").head(10))
```

**What you expect to see:** The age band 17-21 combined with vehicle group 41-50 should appear near the top of the underpredicting list (A/E > 1.0). The main GLM does not know that this combination should be penalised extra hard. You planted this interaction in the data — now you can see where the GLM is bleeding deviance.