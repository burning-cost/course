## Part 14: Connecting to the rate optimiser

In Module 7 you used the `rate-optimiser` library, which required a demand callable as one of its inputs. The `ConversionModel` from `insurance-demand` provides exactly this.

This part shows how to connect the demand model to the rate optimiser so that the Module 7 pipeline uses a properly calibrated elasticity rather than a manually specified coefficient.

### Exporting the conversion model as a demand callable

```python
%md
## Part 14: Rate optimiser integration
```

```python
# The conversion model can be exported as a callable compatible with
# rate-optimiser's DemandModel interface

demand_fn = conv_catboost.as_demand_callable()

# Test it: takes price_ratio (numpy array) and features (Polars DataFrame)
# returns conversion probability for each row
test_features = df_quotes.select(["age", "vehicle_group", "ncd_years",
                                   "area", "channel", "technical_premium"]).head(100)
test_price_ratio = np.full(100, 1.1)  # 10% loading above technical

test_probs = demand_fn(test_price_ratio, test_features)
print(f"Conversion probs at 1.1x tech premium: {test_probs.mean():.4f}")

test_price_ratio_low = np.full(100, 0.95)  # 5% discount
test_probs_low = demand_fn(test_price_ratio_low, test_features)
print(f"Conversion probs at 0.95x tech premium: {test_probs_low.mean():.4f}")
print(f"Effect of 15pp price increase: {(test_probs.mean() - test_probs_low.mean()):.4f}")
```

The conversion probability at 0.95x technical premium (a 5% discount) is higher than at 1.1x (a 10% loading). This is the demand function: lower price, higher conversion. The callable wraps the full CatBoost model including all the feature effects.

### Using the callable in the rate optimiser

In Module 7, you passed `make_logistic_demand(params)` to the rate optimiser. You can replace this with the callable from the trained conversion model:

```python
# The rate-optimiser expects a callable with signature:
# f(price_ratio: np.ndarray, features: pl.DataFrame) -> np.ndarray

# demand_fn from conv_catboost has exactly this signature.
# In a Module 7 notebook, you would use:

# from rate_optimiser import RateChangeOptimiser
# opt = RateChangeOptimiser(
#     policy_data=policy_data,
#     factor_structure=factor_structure,
#     demand_model=demand_fn,   # <-- use the trained model here
# )

print("demand_fn signature matches rate-optimiser's DemandModel interface.")
print("Use conv_catboost.as_demand_callable() in place of make_logistic_demand().")
```

The key advantage is that the demand model now reflects your actual observed data rather than a manually specified elasticity. The rate optimiser will find factor adjustments that are consistent with how your customers actually respond to price changes, not with a benchmark from a generic industry study.