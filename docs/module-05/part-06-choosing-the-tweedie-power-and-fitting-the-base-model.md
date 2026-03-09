## Part 6: Choosing the Tweedie power and fitting the base model

### The Tweedie family

CatBoost's Tweedie loss function models the compound Poisson-Gamma distribution that characterises aggregate insurance losses. The `variance_power` parameter `p` controls the variance-to-mean relationship:

- `p = 1`: Poisson (variance proportional to mean). Appropriate for claim counts only.
- `p = 2`: Gamma (variance proportional to mean squared). Appropriate for claim severity only.
- `1 < p < 2`: Compound Poisson-Gamma. **This is the distribution of aggregate losses** - a point mass at zero (no claims) combined with a positive continuous distribution when claims occur.

For UK motor pure premiums, `p = 1.5` is the standard choice. This sits in the middle of the compound Poisson-Gamma range and reflects both the frequency structure (lots of zeros from no-claim policies) and the severity structure (right-skewed losses when claims occur). Using p=1.3 or p=1.7 makes a small difference to the fit. Using p outside the range (1, 2) makes a large difference and is inappropriate for aggregate loss data.

**Practical note:** if your book has a very low claims rate (e.g. liability, where most policies never claim), you might choose p closer to 1.0. If severity is the dominant driver of variation (e.g. catastrophe-exposed property), p closer to 2.0 is more appropriate. For standard UK motor, 1.5 is correct.

### Build the Pool objects and fit the model

In a new cell:

```python
%md
## Part 6: Training the base Tweedie model
```

```python
# CatBoost Pool objects package features, labels, and metadata together
train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
cal_pool   = Pool(X_cal,   y_cal,   cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

tweedie_params = {
    "loss_function":    "Tweedie:variance_power=1.5",
    "eval_metric":      "Tweedie:variance_power=1.5",
    "learning_rate":    0.05,
    "depth":            5,
    "min_data_in_leaf": 50,    # prevents overfitting to small insurance cells
    "iterations":       500,
    "random_seed":      42,
    "verbose":          100,   # print progress every 100 trees
}

model = CatBoostRegressor(**tweedie_params)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

# Sanity-check predictions on the test set
preds_test = model.predict(test_pool)
print(f"\nTest set predictions:")
print(f"  Min: {preds_test.min():.2f}")
print(f"  Median: {np.median(preds_test):.2f}")
print(f"  Mean: {preds_test.mean():.2f}")
print(f"  Max: {preds_test.max():.2f}")
print(f"  Actual mean pure premium: {y_test.mean():.2f}")
```

**What this does:** creates the three CatBoost Pool objects, sets the Tweedie hyperparameters, and trains the model using the calibration set as an early-stopping validation set. The `verbose=100` setting prints the loss every 100 iterations.

**Why `min_data_in_leaf=50`:** without a minimum leaf size, deep trees can split on cells with only a handful of observations. These splits produce very precise but unreliable predictions for thin-cell risks. Thin-cell risks are exactly where conformal intervals will be widest - we need stable base model predictions in those regions, not wildly varying predictions from overfit splits.

**A note on early stopping:** using the calibration pool for early stopping means the model's iteration count has been influenced by the calibration data. This introduces a very minor dependency. In practice the effect on coverage is negligible. However, if you need strict separation for a regulatory audit, use a separate validation pool (drawn from the training set, not the calibration set) for early stopping and keep the calibration set entirely unseen during model fitting.

**What you should see:** training output like this, with the Tweedie loss printed every 100 iterations:

```python
0:      learn: 2.05xxx  test: 2.07xxx   best: 2.07xxx (0)   total: ...
100:    learn: 1.82xxx  test: 1.85xxx   best: 1.84xxx (87)  total: ...
...
Stopped by early stopping after xxx iterations
```

The test set mean prediction should be close to (but not identical to) the actual mean pure premium. A large discrepancy here would suggest a misconfigured model.