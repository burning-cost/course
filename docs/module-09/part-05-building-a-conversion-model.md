## Part 5: Building a conversion model

We start with the simplest possible demand model: a logistic regression of conversion on price. This tells us the expected conversion rate at current prices for any segment. It is not the causal elasticity - we will address that in Part 7 - but it is the right place to start.

### Why start with logistic regression

There is a principle in modelling that is worth stating explicitly: start simple, then complicate. The logistic regression gives you interpretable coefficients, runs in seconds, and gives you a quick sanity check on the data. If the logistic model gives nonsensical results, there is a data problem that needs fixing before you run anything more complicated.

CatBoost will give better predictive accuracy on held-out data. But the logistic model is where you understand what is driving the predictions.

### Fitting the baseline logistic conversion model

Add a markdown cell:

```python
%md
## Part 5: Logistic conversion model
```

Then:

```python
conv_logistic = ConversionModel(
    base_estimator="logistic",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
    logistic_C=1.0,
)
conv_logistic.fit(df_quotes)

# Coefficients
print(conv_logistic.summary())
```

The `summary()` method for the logistic backend returns a DataFrame with the coefficient for each feature and the odds ratio (exp(coefficient)). The most important row is `log_price_ratio`, the log of the quoted price relative to the technical premium.

You should see a coefficient around -1.5 to -2.5 for `log_price_ratio`. This is the logistic model's estimated price effect - but remember, it includes confounding from the risk composition. We will compare it to the DML estimate later.

Now compute predicted conversion rates:

```python
conv_probs = conv_logistic.predict_proba(df_quotes)
print(f"Mean predicted conversion:  {conv_probs.mean():.3f}")
print(f"Mean observed conversion:   {df_quotes['converted'].mean():.3f}")

# Overall calibration: the mean should match since we're predicting in-sample
# The important check is calibration within segments
```

### One-way observed vs. fitted plots

The one-way plot is the pricing actuary's standard model diagnostic. For each level of a rating factor, it compares the observed conversion rate to the fitted conversion rate. A good model tracks the observed closely; gaps indicate either missing interactions or model misspecification.

```python
# One-way by channel
channel_ow = conv_logistic.oneway(df_quotes, "channel")
print("One-way by channel:")
print(channel_ow.to_string())
```

```python
# One-way by NCD years
ncd_ow = conv_logistic.oneway(df_quotes, "ncd_years")
print("One-way by NCD years:")
print(ncd_ow.to_string())
```

Look at the `lift` column in each table. Lift is observed_rate / fitted_rate. Values near 1.0 mean the model is well-calibrated for that group. Values above 1.2 or below 0.8 indicate the model is under- or over-predicting for that segment.

PCW channels should show higher conversion rates than direct at the same price ratio, because PCW customers are actively shopping and have already shown intent. If your PCW one-way shows lift far from 1.0 for all channels, the channel feature is not being captured properly.

```python
# One-way by age (binned)
age_ow = conv_logistic.oneway(df_quotes, "age", bins=10)
print("One-way by age decile:")
print(age_ow.to_string())
```

### Upgrading to CatBoost

Once the logistic model looks reasonable, fit the CatBoost version. This will take 30-60 seconds on Databricks Free Edition.

```python
conv_catboost = ConversionModel(
    base_estimator="catboost",
    feature_cols=["age", "vehicle_group", "ncd_years", "area", "channel"],
    rank_position_col="rank_position",
    cat_features=["area", "channel"],  # tell CatBoost these are categorical
)
conv_catboost.fit(df_quotes)

print("CatBoost feature importances:")
print(conv_catboost.summary())
```

The CatBoost backend does not give you coefficients (it is a non-linear tree model). Instead, `summary()` returns feature importances - which features the model relies on most for its predictions. You would typically expect `log_price_ratio` and `log_rank` (the log of the PCW rank position) to be among the top features.

Compare predictive accuracy between the two models:

```python
from sklearn.metrics import roc_auc_score

y_true = df_quotes["converted"].to_numpy()

auc_logistic = roc_auc_score(y_true, conv_logistic.predict_proba(df_quotes).to_numpy())
auc_catboost = roc_auc_score(y_true, conv_catboost.predict_proba(df_quotes).to_numpy())

print(f"Logistic AUC: {auc_logistic:.4f}")
print(f"CatBoost AUC: {auc_catboost:.4f}")
```

CatBoost will typically show a Gini around 0.65-0.75 on this data versus 0.60-0.68 for logistic. The improvement is real but not dramatic - the price ratio and rank position dominate conversion, and both models capture those.

For production use, you would do this comparison on a held-out test set (policies from a later date period, not a random sample). The module on model monitoring (Module 11) covers the walk-forward validation approach. For now, in-sample AUC is sufficient to confirm the models are working.

### Why the logistic coefficient is still biased

Before moving to DML, confirm the bias in the logistic coefficient. In the summary table, find the coefficient on `log_price_ratio`. Compare it to the true elasticity stored in the dataset:

```python
# True population-average elasticity
true_elas = df_quotes["true_elasticity"].mean()
print(f"True elasticity:           {true_elas:.3f}")

# Logistic model's implied price coefficient
summary = conv_logistic.summary()
price_coef = summary.loc[summary["feature"] == "log_price_ratio", "coefficient"].values[0]
print(f"Logistic price coefficient: {price_coef:.3f}")
print(f"Bias: {abs(price_coef - true_elas):.3f}")
```

You should see that the logistic coefficient is more negative than the true elasticity. This is the confounding bias: the model is attributing some of the risk-composition effect to the price effect, making the book look more elastic than it is. If you used this coefficient to optimise prices, you would under-price inelastic segments and over-price elastic ones.