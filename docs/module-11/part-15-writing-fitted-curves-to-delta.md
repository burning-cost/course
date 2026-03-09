## Part 15: Writing fitted curves to Delta

Like any modelling output, the fitted distribution parameters should be persisted with a full audit trail. On Databricks, this means writing to a Delta table.

```python
%md
## Part 15: Persisting fitted curves to Delta
```

```python
# Persist the fitted curve parameters and goodness-of-fit diagnostics
fit_record = pl.DataFrame({
    "fit_date":       [str(np.datetime64("today"))],
    "curve_name":     ["commercial_property_fitted"],
    "g":              [fitted_dist.g],
    "b":              [fitted_dist.b],
    "total_loss_prob":[fitted_dist.total_loss_prob()],
    "mean_z":         [fitted_dist.mean()],
    "loglik":         [result.loglik],
    "aic":            [result.aic],
    "bic":            [result.bic],
    "n_obs":          [result.n_obs],
    "method":         [result.method],
    "converged":      [result.converged],
    "ks_statistic":   [gof.ks_test()["statistic"]],
    "ks_pvalue":      [gof.ks_test()["p_value"]],
    "ad_statistic":   [gof.ad_test()["statistic"]],
})

print("Curve fitting record:")
print(fit_record)

# On Databricks, write to Delta:
# spark.createDataFrame(fit_record.to_pandas()).write.format("delta").mode("append") \
#     .save("dbfs:/user/your_user/module11/fitted_curves")

# For local development (no Spark), save as Parquet:
fit_record.write_parquet("/tmp/fitted_curves.parquet")
print("\nSaved to /tmp/fitted_curves.parquet")
```

### Why provenance matters

Every ILF table used in a London market submission should be traceable back to the data and parameters that produced it. The Delta record above provides:
- The date of fitting
- The fitted parameters (g, b) -- from which the ILF table can be exactly reproduced
- Goodness-of-fit statistics -- evidence the curve was checked, not just accepted
- The number of observations and fitting method -- evidence of the data source

Lloyd's manages this through the Model Registration Form (MRF) process for internal models. Non-Lloyd's commercial lines actuaries face a similar auditability requirement under the FCA's Senior Managers and Certification Regime. "We used Y2 because that's what we've always used" is not an adequate response to a supervisor's enquiry. "We fitted the MBBEFD distribution to 600 claims with KS p-value 0.82 and AIC 628.8, with the parameters g=22.43, b=3.18 stored in Delta table curve_registry/commercial_property_fitted" is.