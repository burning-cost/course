## Part 5: The three-way temporal split

This is the most important structural decision in the module. The split determines whether the conformal coverage guarantee is valid.

### Why three sets, not two

Standard machine learning uses two sets: training (to fit the model) and test (to evaluate it). Conformal prediction needs three:

1. **Training set** - the model learns the relationship between features and losses
2. **Calibration set** - the conformal predictor measures how well the model's predictions are calibrated; this set is entirely unseen during model training
3. **Test set** - we evaluate whether the intervals produced using the calibration quantile actually achieve the stated coverage on new data

The calibration set must be unseen during model training. If any calibration observation influenced the model's parameters, the coverage guarantee can fail - the model may have partially memorised those observations, producing artificially small residuals and a calibration quantile that is too optimistic.

### Why temporal, not random

The exchangeability assumption underpinning the coverage guarantee requires that calibration data comes from the same distribution as test data. In insurance, time matters: claim frequencies, severities, and the mix of risks change from year to year. A random split where calibration data includes policies from 2020 and test data also includes policies from 2020 treats them as if they came from the same "snapshot," which hides temporal trends in the residuals.

The temporal split - training on older data, calibrating on middle data, testing on the most recent data - is the correct approach. It matches the operational reality: you train on historical business, calibrate on recent business, and make predictions for future business.

In a new cell, type this and run it:

```python
%md
## Part 5: Temporal data split
```

```python
# The DataFrame is already sorted by accident_year (we did this above)
n   = len(df)
X_COLS = ["vehicle_group", "driver_age", "ncd_years", "area", "conviction_points", "annual_mileage"]
CAT_FEATURES = ["area"]

train_end = int(0.60 * n)   # first 60% of rows (by accident year) = training
cal_end   = int(0.80 * n)   # next 20% = calibration

# Training set: oldest 60% of policies
X_train = df[:train_end][X_COLS].to_pandas()
y_train = df[:train_end]["pure_premium"].to_pandas()
e_train = df[:train_end]["exposure"].to_numpy()

# Calibration set: next 20%
X_cal   = df[train_end:cal_end][X_COLS].to_pandas()
y_cal   = df[train_end:cal_end]["pure_premium"].to_pandas()

# Test set: most recent 20%
X_test  = df[cal_end:][X_COLS].to_pandas()
y_test  = df[cal_end:]["pure_premium"].to_pandas()

# Verify the temporal split is what we think it is
train_years = df[:train_end]["accident_year"].unique().sort()
cal_years   = df[train_end:cal_end]["accident_year"].unique().sort()
test_years  = df[cal_end:]["accident_year"].unique().sort()

print(f"Training set:     {len(X_train):,} rows, years: {train_years.to_list()}")
print(f"Calibration set:  {len(X_cal):,}  rows, years: {cal_years.to_list()}")
print(f"Test set:         {len(X_test):,}  rows, years: {test_years.to_list()}")
```

**What this does:** splits the sorted DataFrame at the 60% and 80% marks. Because we sorted by `accident_year` before splitting, the training set contains the oldest policies, the calibration set contains the next-most-recent, and the test set contains the most recent.

**What you should see:** three sets of roughly 20,000, 20,000, and 20,000 rows (they will not be exactly equal because accident years are not uniformly distributed). The years should not overlap between sets - if they do, the sort did not work.

```
Training set:     60,xxx rows, years: [2019, 2020, 2021, 2022]
Calibration set:  20,xxx rows, years: [2022, 2023]
Test set:         20,xxx rows, years: [2023, 2024]
```

There will be some year overlap at the boundaries (e.g. 2022 appears in both training and calibration) because the rows within a year are ordered by `rng.choice` rather than strictly chronologically. This is acceptable - the important thing is that the majority of calibration rows are more recent than the majority of training rows.