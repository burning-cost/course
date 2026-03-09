## Part 6: Why random cross-validation is wrong for insurance data

This is the section that most pricing teams get wrong, and getting it wrong leads to over-confident deployment decisions.

Standard machine learning practice is an 80/20 random train/test split. For insurance data, this is wrong. It produces optimistic performance estimates that overstate real out-of-sample performance.

The reason is temporal autocorrelation. Insurance claims data has structure across accident years: inflation trends, frequency cycles, mix changes. A model trained on a random 80% of 2019-2024 data sees policies from all six years in both training and test. When it makes predictions on the test set, it is not generalising to new data - it is filling in gaps in a history it has already seen. The temporal patterns in the test set match those in the training set, so the model looks better than it will be when predicting 2025 policies from 2024 training data.

Walk-forward cross-validation replicates the actual deployment scenario:

- Fold 1: train on 2019-2020, validate on 2022 (2021 excluded as IBNR buffer)
- Fold 2: train on 2019-2021, validate on 2023 (2022 excluded as IBNR buffer)
- Fold 3: train on 2019-2022, validate on 2024 (2023 excluded as IBNR buffer)

**What the IBNR buffer does:** Claims from the most recent accident year are typically 30-50% underdeveloped at the time you would be fitting the model. A claim from December 2022 accident year may not be fully settled until late 2023 or 2024. Including that year in training gives the model access to partially-developed claims data that will not be available at deployment time. Excluding the year immediately before the validation year is more conservative - and more honest about what the model can actually do in production.

### Setting up the walk-forward CV

The `insurance-cv` library generates these splits automatically. Create a new cell:

```python
# Convert to pandas for the CV split - WalkForwardCV uses pandas index arrays
df_pd = df.to_pandas()

cv = WalkForwardCV(
    year_col="accident_year",
    min_train_years=2,
    ibnr_buffer_years=1,
    n_splits=3,
)

folds = list(cv.split(df_pd))

# Print the fold structure so you can see exactly what is being trained and validated
for i, (train_idx, val_idx) in enumerate(folds):
    train_years = sorted(df_pd.iloc[train_idx]["accident_year"].unique().tolist())
    val_years   = sorted(df_pd.iloc[val_idx]["accident_year"].unique().tolist())
    print(f"Fold {i+1}: train years = {train_years}, validate years = {val_years}")
```

Run this. You should see output like:

```sql
Fold 1: train years = [2019, 2020], validate years = [2022]
Fold 2: train years = [2019, 2020, 2021], validate years = [2023]
Fold 3: train years = [2019, 2020, 2021, 2022], validate years = [2024]
```

**Why 2021 is missing from Fold 1:** It is the IBNR buffer year between the training data (ending 2020) and the validation year (2022). Same for 2022 in Fold 2, and 2023 in Fold 3.

The `folds` variable contains a list of (train_index, val_index) pairs. These are the row indices into `df_pd` that belong to each split. You will use them in the CV loop in Part 8.