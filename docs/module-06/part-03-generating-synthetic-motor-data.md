## Part 3: Generating synthetic motor data

We use synthetic data so we know the true underlying risk — this lets us verify that credibility estimation is working correctly. Real data validation is covered in Exercise 1.

### The data generating process

Create a new cell with a markdown header:

```python
%md
## Part 3: Synthetic motor portfolio — postcode districts
```

Create the next cell:

```python
rng = np.random.default_rng(seed=42)

# Portfolio parameters
N_DISTRICTS = 120          # 120 UK postcode districts
N_YEARS = 5                # accident years 2019-2023
PORTFOLIO_FREQUENCY = 0.07 # 7% mean claim frequency (motor)

# True between-district heterogeneity on log scale
# This is the ground truth sigma_district - we will try to recover it
TRUE_SIGMA_DISTRICT = 0.35

# True district-level log-rate deviations from the portfolio mean
# These are the "true" risk levels we are trying to estimate
true_log_rates = rng.normal(0, TRUE_SIGMA_DISTRICT, size=N_DISTRICTS)

# Postcode district names: mix of real UK formats
prefixes = ["SW", "SE", "N", "E", "W", "EC", "WC", "KT", "SM", "CR",
            "BR", "DA", "RM", "EN", "HA", "UB", "TW", "IG", "WD", "SL",
            "GU", "RH", "TN", "ME", "CT", "BN", "PO", "SO", "SP", "RG",
            "OX", "MK", "NN", "LE", "PE", "CB", "IP", "NR", "CO", "CM",
            "SS", "RM", "AL", "LU", "SG", "HP", "SN", "BA", "BS", "GL",
            "HR", "WR", "B", "CV", "DE", "NG", "LN", "HU", "DN", "S",
            "HD", "HX", "BD", "LS", "WF", "WA", "SK", "ST", "WV", "DY",
            "TF", "SY", "LL", "SA", "CF", "NP", "LD", "SY", "LA", "FY",
            "PR", "BB", "OL", "M", "BL", "WN", "L", "CH", "CW", "CA",
            "DL", "HG", "YO", "TS", "NE", "DH", "SR", "TD", "EH", "G",
            "PA", "KA", "FK", "KY", "DD", "PH", "AB", "IV", "KW", "HS"]

# Ensure we have exactly N_DISTRICTS names
district_names = []
for i, prefix in enumerate(prefixes[:N_DISTRICTS]):
    district_names.append(f"{prefix}{i+1}")

print(f"Generated {len(district_names)} district names")
print(f"First 10: {district_names[:10]}")
```

**What this does:** Sets up the ground-truth parameters for the simulation. `TRUE_SIGMA_DISTRICT = 0.35` means the true between-district log-rate standard deviation is 0.35. On the claim rate scale, this means districts range from roughly `exp(-0.35) = 0.70x` to `exp(+0.35) = 1.42x` the portfolio mean at ±1 SD. A significant amount of genuine geographic heterogeneity — typical for UK motor.

**Run this cell.**

**What you should see:** A count of 120 district names and the first 10. The district names look like real UK postcode prefixes with a number appended.

Now create the data for each district and year:

```python
# Generate exposures: highly skewed — inner city districts are dense,
# rural districts are thin. Matches the real distribution of UK motor books.
base_exposures = rng.lognormal(mean=5.5, sigma=1.2, size=N_DISTRICTS)
base_exposures = np.clip(base_exposures, 20, 5000)

# Create a DataFrame with one row per (district, accident_year)
rows = []
for i, district in enumerate(district_names):
    for year in range(2019, 2019 + N_YEARS):
        # Exposure varies slightly by year (lapse and new business fluctuation)
        exposure_this_year = base_exposures[i] * rng.uniform(0.85, 1.15)

        # True log-rate: portfolio base + district effect (stable over time)
        # Real insurance: districts are also affected by portfolio-wide trends,
        # but for simplicity we hold the district effect fixed
        true_log_rate = np.log(PORTFOLIO_FREQUENCY) + true_log_rates[i]
        true_rate = np.exp(true_log_rate)

        # Observed claims: Poisson given true rate and exposure
        observed_claims = rng.poisson(true_rate * exposure_this_year)

        rows.append({
            "postcode_district": district,
            "accident_year": year,
            "earned_years": float(exposure_this_year),
            "claim_count": int(observed_claims),
            "true_rate": float(true_rate),
            "true_log_deviation": float(true_log_rates[i]),
        })

df = pl.DataFrame(rows)

# Compute observed claim frequency per row
df = df.with_columns(
    (pl.col("claim_count") / pl.col("earned_years")).alias("claim_frequency")
)

print(f"Dataset dimensions: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))
```

**What this does:** Generates a panel dataset: 120 districts × 5 years = 600 rows. Each row has the number of claims observed and the earned exposure (policy-years) in that district-year combination. The `true_rate` column contains the ground truth — what the credibility estimator should try to recover. We keep this column in the DataFrame for validation; in real data, of course, you would not have it.

**Run this cell.**

**What you should see:** `Dataset dimensions: (600, 7)` and a table showing district names, accident years, earned years (anywhere from ~20 to ~5,000), claim counts, and the true rate. Look at the `earned_years` column — you should see a large spread. Some districts have hundreds of policy-years per year; others have 20-30. This is realistic.

Now compute district-level aggregate statistics, which are what the credibility estimator uses:

```python
# Aggregate to district level (summing across years)
dist_totals = (
    df
    .group_by("postcode_district")
    .agg([
        pl.col("earned_years").sum().alias("total_earned_years"),
        pl.col("claim_count").sum().alias("total_claims"),
        pl.col("true_rate").mean().alias("true_rate"),        # the ground truth
        pl.col("true_log_deviation").mean().alias("true_log_deviation"),
    ])
    .with_columns([
        (pl.col("total_claims") / pl.col("total_earned_years")).alias("observed_frequency"),
    ])
    .sort("postcode_district")
)

print(f"Number of districts: {dist_totals.height}")
print()
print("Exposure distribution across districts:")
print(dist_totals["total_earned_years"].describe())
print()
print("Top 5 thinnest districts (fewest policy-years):")
print(dist_totals.sort("total_earned_years").head(5))
print()
print("Top 5 densest districts (most policy-years):")
print(dist_totals.sort("total_earned_years", descending=True).head(5))
```

**What this does:** Creates a district-level summary with the total exposure and total claims across all five years. This is the starting point for Bühlmann-Straub credibility — you need group-level totals and the within-group year-by-year variation.

**Run this cell.**

**What you should see:** 120 districts. The exposure distribution will be highly right-skewed — a few dense districts with 5,000-25,000 earned years across 5 years, and many thin districts with 100-500 earned years. The thinnest districts will have observed frequencies that look implausible — a district with 20 claims across 5 years of 50 policy-years each will have an observed frequency that is highly volatile.

### Checkpoint 1: Data check

Before proceeding, verify your dataset is sensible:

```python
# Checkpoint 1: Basic sanity checks on the data
total_policies = dist_totals["total_earned_years"].sum()
total_claims = dist_totals["total_claims"].sum()
portfolio_frequency = total_claims / total_policies

print("=== CHECKPOINT 1: DATA SANITY ===")
print(f"Total earned years (portfolio): {total_policies:,.0f}")
print(f"Total claims:                   {total_claims:,}")
print(f"Portfolio claim frequency:       {portfolio_frequency:.4f}  ({portfolio_frequency*100:.2f}%)")
print()
print(f"Expected portfolio frequency:    {PORTFOLIO_FREQUENCY:.4f}  ({PORTFOLIO_FREQUENCY*100:.2f}%)")
print()

# Check for problematic rows
zero_exposure = dist_totals.filter(pl.col("total_earned_years") == 0).height
zero_claims = dist_totals.filter(pl.col("total_claims") == 0).height
print(f"Districts with zero exposure:  {zero_exposure}  (should be 0)")
print(f"Districts with zero claims:    {zero_claims}  (may be non-zero for thin districts)")
print()

if abs(portfolio_frequency - PORTFOLIO_FREQUENCY) / PORTFOLIO_FREQUENCY < 0.05:
    print("Portfolio frequency is within 5% of target. Proceeding.")
else:
    print("WARNING: Portfolio frequency deviates by more than 5%. Check data generation.")
```

**What you should see:** The portfolio frequency close to 7% (within 5%). Zero rows with zero exposure. Possibly a handful of districts with zero claims (these are the very thinnest districts that happened to have no claims in five years — they are real and legitimate).

If you see warnings here, re-read the data generation cell above and run it again. The random seed is fixed at 42, so results should be reproducible.