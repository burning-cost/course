## Part 6: Building a retention model

The retention model is the renewal equivalent of the conversion model. It predicts the probability that an existing customer renews, given their current features and the price change from last year.

The treatment variable is different from conversion. For retention, what the customer reacts to is not the absolute price but the change from what they paid before. A customer paying £600 who is offered £630 sees a 5% increase. Whether that feels expensive depends on their expectations, their tenure, and the market alternatives. The standard treatment is `log(renewal_price / prior_year_price)`.

### Fitting the logistic retention model

```python
%md
## Part 6: Retention model
```

```python
retention_model = RetentionModel(
    model_type="logistic",
    outcome_col="lapsed",
    price_change_col="log_price_change",
    feature_cols=["tenure_years", "ncd_years", "payment_method",
                  "age", "channel", "region"],
    cat_features=["payment_method", "channel", "region"],
)

# The retention dataset uses 'lapsed' as the outcome (1 = lapsed, 0 = renewed)
# We need to ensure this column exists. The make_renewal_data dataset uses 'renewed'.
# Add a lapsed column:
df_renewals_with_lapsed = df_renewals.with_columns(
    (1 - pl.col("renewed")).alias("lapsed")
)

retention_model.fit(df_renewals_with_lapsed)

lapse_probs = retention_model.predict_proba(df_renewals_with_lapsed)
print(f"Mean predicted lapse rate: {lapse_probs.mean():.3f}")
print(f"Mean observed lapse rate:  {df_renewals_with_lapsed['lapsed'].mean():.3f}")
```

Now look at the model summary to understand which factors drive lapse:

```python
print(retention_model.summary())
```

Interpret the coefficients. You should see:

- `log_price_change`: negative coefficient on lapse (higher price change = more lapse), positive on renewal. This is the price effect.
- `tenure_years`: negative on lapse. Longer-tenured customers are stickier.
- `payment_method`: direct debit customers lapse less than annual payers. This is a well-established empirical finding in UK motor: people who pay monthly by DD face friction to cancel.
- `ncd_years`: higher NCD customers lapse less, because they value their NCD protection and know it is insurer-specific.

### Predicting renewal probability

```python
# The model predicts lapse probability. Renewal probability is the complement.
renewal_probs = retention_model.predict_renewal_proba(df_renewals_with_lapsed)
print(f"Mean predicted renewal prob: {renewal_probs.mean():.3f}")
print(f"Observed renewal rate:       {df_renewals['renewed'].mean():.3f}")
```

### Sensitivity analysis: how much does price change matter?

```python
# Price sensitivity: dP(lapse)/d(log_price_change) per segment
sensitivity = retention_model.price_sensitivity(df_renewals_with_lapsed)
print(f"Mean price sensitivity: {sensitivity.mean():.4f}")
print(f"  (negative = higher price → more lapse)")
```

Now look at this by channel - the segmentation that matters most for commercial decisions:

```python
# Combine sensitivity with channel information using Polars
sensitivity_pl = (
    df_renewals
    .select(["channel", "ncd_years"])
    .with_columns(pl.Series("sensitivity", sensitivity.values))
    .group_by("channel")
    .agg(pl.col("sensitivity").mean().alias("mean_sensitivity"))
    .sort("mean_sensitivity")
)

print("Mean price sensitivity by channel:")
print(sensitivity_pl)
```

PCW customers should show higher price sensitivity (more negative) than direct customers. Broker customers (if present) should be the least sensitive - they have an advisor relationship that creates switching friction.

### When to use a CatBoost retention model

The logistic retention model is the right default. It is interpretable, calibratable, and understood by compliance and actuarial governance. Use a CatBoost retention model when:

1. You have a large book (50k+ policies) and are seeing consistent lift problems in the one-way diagnostics
2. The interaction between tenure and channel is important and the logistic model is not capturing it
3. You are optimising for accuracy of predicted lapse rates by segment rather than interpretability

To fit a CatBoost retention model, change `model_type="catboost"` and add `cat_features=["payment_method", "channel", "region"]`. The fit takes longer but the API is identical. We do not run it here to save time on Databricks Free Edition.