## Part 7: The CatBoost Pool object

Before we get to the CV loop, we need to understand the `Pool` object. It is CatBoost's way of bundling together the features, the target, and any metadata (exposure offset, categorical feature list) so you do not have to pass them separately to every function call.

A Pool looks like this:

```python
from catboost import CatBoostRegressor, Pool

train_pool = Pool(
    data=X_train,           # feature matrix (DataFrame or numpy array)
    label=y_train,          # target column (claim counts)
    baseline=np.log(w_train),  # exposure offset: log(exposure_years)
    cat_features=CAT_FEATURES, # list of column names that are categorical
)
```

The Pool pre-processes the categorical features using CatBoost's ordered target statistics. It does this work once at Pool construction time rather than on every training iteration. This is why you should always build the Pool once and reuse it - constructing it inside a loop wastes the encoding work.

Once you have a Pool, you pass it directly to `model.fit()` and `model.predict()`:

```python
model.fit(train_pool, eval_set=val_pool)  # eval_set is the validation Pool
predictions = model.predict(val_pool)      # returns predicted counts (not frequencies)
```

**What `.predict()` returns:** The CatBoost Poisson model predicts expected claim counts, not expected claim frequencies. A policy with exposure 0.5 and a frequency of 0.06 gets a predicted count of 0.03. To convert to frequency, divide by exposure: `predicted_count / exposure`.