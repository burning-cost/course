## Part 16: Adapting the pipeline for production

The notebook runs end-to-end on synthetic data. Four changes adapt it to a real motor book.

### Stage 2: Replace data generation with a real data read

```python
# Replace the cohort generation loop in Stage 2 with:
raw_pl = pl.from_pandas(
    spark.table("source_system.motor.policy_claims_view")
    .filter("accident_year between 2021 and 2025")
    .filter("policy_status = 'in-force' or policy_status = 'expired'")
    .toPandas()
)
```

The only requirement is that the input DataFrame has the columns that `apply_transforms()` expects. For the current transform list: `age_band`, `ncb`, `vehicle_group`, `region`, `annual_mileage`, `exposure`, `claim_count`, `incurred_loss`, `accident_year`.

If your source data uses different column names, add a rename step before `apply_transforms()`:

```python
raw_pl = raw_pl.rename({
    "driver_age_band": "age_band",
    "ncd_years":       "ncb",
    "veh_group":       "vehicle_group",
    "region_code":     "region",
    "annual_km":       "annual_mileage",
    "earned_exp":      "exposure",
    "claims_n":        "claim_count",
    "incurred":        "incurred_loss",
})
```

Do not add the renaming logic inside `apply_transforms()`. Keep `apply_transforms()` as a pure feature engineering function; column renaming is data preparation and belongs in Stage 2.

### Stage 3: Update the transform list for your feature set

Add new transform functions to the TRANSFORMS list. Each function takes a Polars DataFrame and returns a Polars DataFrame with additional columns. Update FEATURE_COLS to include the new columns.

```python
# Example: add a young driver × high vehicle group interaction
def add_young_high_vg(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("age_band") == "17-25") &
            (pl.col("vehicle_ord") >= 4)
        ).cast(pl.Int32).alias("young_high_vg")
    )

TRANSFORMS   = [encode_ncb, encode_vehicle, encode_age, encode_mileage,
                add_log_exposure, add_young_high_vg]
FEATURE_COLS = ["ncb_deficit", "vehicle_ord", "age_mid", "mileage_ord",
                "region", "young_high_vg"]
```

Whenever you modify TRANSFORMS or FEATURE_COLS, re-run from Stage 3 onwards. The CV metrics will reflect the new feature set and will be comparable to any previous run that used the same base data.

### Stage 4: Adjust the IBNR buffer for your line of business

For the `insurance-cv` version:

```python
ibnr = IBNRBuffer(months=12)   # motor BI: 12-18 months
# or
ibnr = IBNRBuffer(months=24)   # employers' liability, solicitors' PI: 24+ months
```

For the manual fold version, trim the trailing N months from the training set in each fold:

```python
# Trim last 12 months from each training fold
IBNR_BUFFER_MONTHS = 12
cutoff_date = df_tr["accident_month"].max() - pd.DateOffset(months=IBNR_BUFFER_MONTHS)
df_tr = df_tr[df_tr["accident_month"] <= cutoff_date]
```

This requires monthly accident period data. If you only have annual accident years, the buffer has limited effect — you can exclude the most recent accident year from training, which is equivalent to a 6-12 month buffer depending on the extract date.

### Stage 5: Increase Optuna trials for production runs

```python
N_OPTUNA_TRIALS = 40   # up from 20 for the tutorial
```

Forty trials per model adds roughly 15-20 minutes of compute on a 4-core cluster. For a quarterly review cycle where the pipeline runs once per quarter, this is negligible. The marginal improvement over 20 trials is typically 0.001-0.003 deviance units — small in absolute terms but meaningful for portfolios of 100,000+ policies where a 0.001 improvement in deviance corresponds to thousands of pounds of pricing precision.

### Stage 9: Production renewal portfolio inputs

In production, the renewal portfolio comes directly from your policy administration system:

```python
renewal_pd = spark.table("source_system.motor.renewals_next_quarter").toPandas()

# Required columns for PortfolioOptimiser:
# - technical_price (from freq * sev model predictions)
# - expected_loss_cost (same as technical_price in a well-calibrated model)
# - p_demand (from your retention model, ideally estimated from lapse data)
# - elasticity (from your causal price elasticity model — Module 9)
# - renewal_flag (all True for renewals; False for new business)
# - enbp (from your new-business pricing model for equivalent risk)

opt = PortfolioOptimiser(
    technical_price=renewal_pd["tech_prem"].values,
    expected_loss_cost=renewal_pd["expected_loss"].values,
    p_demand=renewal_pd["p_renew"].values,
    elasticity=renewal_pd["elasticity"].values,
    renewal_flag=renewal_pd["is_renewal"].values,
    enbp=renewal_pd["nb_equivalent_prem"].values,
    constraints=config,
)
```

The demand parameters in Stage 9 of this module are synthetic. Replacing them with estimates from your own book's lapse data is the single most important production adaptation — incorrect demand parameters will cause the optimiser to either underprice (overestimating sensitivity) or overprice (underestimating sensitivity) the book. Module 9 covers causal demand estimation from observational data.
