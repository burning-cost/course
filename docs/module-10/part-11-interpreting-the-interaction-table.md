## Part 11: Interpreting the interaction table

Let us look at the table in detail and understand what each column tells us.

```python
# Detailed view of recommended interactions
recommended = table.filter(pl.col("recommended") == True)
print(f"Recommended interactions ({recommended.height} pairs):")
print()

for row in recommended.iter_rows(named=True):
    print(f"{row['feature_1']} × {row['feature_2']}")
    print(f"  NID rank:         {int(row['nid_rank'])}")
    print(f"  NID score:        {row['nid_score_normalised']:.3f}")
    print(f"  Parameter cost:   {row['n_cells']} cells")
    print(f"  Deviance gain:    {row['delta_deviance']:.1f} ({row['delta_deviance_pct']:.2f}%)")
    print(f"  LR chi2:          {row['lr_chi2']:.1f} (df={row['lr_df']})")
    print(f"  p-value:          {row['lr_p']:.2e}")
    print(f"  AIC change:       {row['delta_deviance_aic']:.1f}")
    print(f"  BIC change:       {row['delta_deviance_bic']:.1f}")
    print()
```

**Interpreting the numbers:**

- **NID score** tells you how strong the interaction was in the trained CANN. This is a ranking tool, not a statistical measure.
- **n_cells** is the parameter cost. An interaction with `n_cells = 20` adds 20 new parameters to the GLM. You want the deviance gain to be comfortably larger than `2 × n_cells` (the AIC cost) before adding it.
- **delta_deviance** and **delta_deviance_pct** show how much the fit improves. A 0.3% deviance improvement sounds small but represents a real improvement in model calibration.
- **delta_deviance_aic** must be negative for the interaction to improve AIC (lower is better). If AIC goes up despite a significant LR test, the improvement does not justify the parameter cost.
- **delta_deviance_bic** applies a stricter penalty than AIC (penalises by `log(n)` rather than 2 per parameter). BIC is stricter for large datasets.

Note on the AIC/BIC values: these are *deviance-based* information criteria (D + 2k and D + k·log(n)), not the standard AIC from R's `AIC()` function. The delta values are identical to standard delta-AIC/BIC, but the absolute values differ by a constant (the saturated log-likelihood). Use the delta columns for model comparison; the absolute values are not comparable across software.

### When NOT to add a significant interaction

Statistical significance and practical usefulness are different things. You might choose not to add a recommended interaction if:

1. **n_cells is very high.** A 6-level age band × 50-band vehicle group (non-banded) interaction adds 245 parameters. You need enormous data to estimate 245 interaction cells credibly. High NCD-5 young drivers in vehicle group 43 will appear in very few policies. The interaction term for that specific cell will have a huge standard error.

2. **delta_deviance_aic is positive despite significant LR test.** This happens when `n_cells` is large enough that the AIC cost outweighs the deviance gain. The GLM test says the interaction is real; AIC says the model does not benefit enough from it to justify the complexity.

3. **The interaction makes no underwriting sense.** If `annual_mileage × conviction_points` has a high NID score and a significant LR test but there is no plausible causal mechanism, investigate further. The signal might be a data artefact (e.g., how conviction points were collected, or whether annual_mileage has recording issues for young drivers with convictions).

For thin interaction cells where `n_cells` is high but the cells have sparse data, consider Bayesian regularisation of the interaction parameters rather than dropping them entirely. Module 6's credibility weighting approach applies directly to interaction parameter estimation — shrink the cell-level estimates towards the marginal effects rather than setting them to zero.
