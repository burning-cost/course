## Part 4: Bühlmann-Straub credibility — the maths and the code

### Why this section matters

Bühlmann-Straub (1970) is the workhorse credibility method in European non-life insurance. It appears in the Swiss Solvency Test standard formula, in IFoA working papers, and in the actuarial standards of the Netherlands, Germany, and Switzerland. It is not exotic — it is the established method.

UK actuaries who trained on Emblem or Radar have seen GLM coefficients with standard errors. Bühlmann-Straub is giving you something conceptually similar: a parameter estimate (the credibility-weighted rate) with an implicit uncertainty measure (the credibility factor Z). The difference is that B-S is designed explicitly for the problem of blending group-specific evidence with portfolio evidence, rather than estimating a main effect.

### The three structural parameters

The B-S model operates on a dataset of groups (districts, schemes, vehicle classes — any set of segments) observed over multiple periods. Three numbers drive everything:

**mu (grand mean):** The portfolio-wide expected loss rate, weighted by exposure. This is the anchor — the value a thin district's estimate collapses toward.

**v (EPV — Expected value of Process Variance):** Within-group variance, averaged over the portfolio. This captures how much a group's observed rate fluctuates year to year, purely due to Poisson sampling noise, even if its true underlying risk is perfectly stable. High v means groups are inherently volatile.

**a (VHM — Variance of Hypothetical Means):** Between-group variance. This captures how much the true underlying risks differ across groups. High a means the portfolio is genuinely heterogeneous — some districts really are riskier than others.

**K = v / a:** The credibility parameter. Interpretable as: "how many units of exposure does a group need before its own experience is as informative as the portfolio mean?" Low K means you trust group experience quickly. High K means you need many years of data before the portfolio mean is overridden.

### The credibility factor Z

For group i with total exposure w_i, the credibility factor is:

```sql
Z_i = w_i / (w_i + K)
```

The credibility-weighted estimate is:

```sql
P_i = Z_i × X̄_i + (1 - Z_i) × mu
```

where X̄_i is the exposure-weighted observed mean for group i.

This formula has a clean intuitive reading:
- When w_i is large (many policy-years), Z_i → 1, and the estimate trusts the group's own experience
- When w_i is small, Z_i → 0, and the estimate collapses to the portfolio mean mu
- K controls the speed of this transition

For KT with 847 policy-years and K = 1,200: Z = 847 / (847 + 1200) = 0.41. KT's rate would be 41% of its own observed 1.30% and 59% of the portfolio mean 6.8%. That gives 0.41 × 0.013 + 0.59 × 0.068 = 0.046, i.e. 4.6%. A meaningful adjustment downward from the portfolio mean, but far from KT's own volatile 1.3%.

### Estimating the structural parameters from data

You do not specify v and a in advance — they are estimated from the data. Here are the formulas, followed immediately by the implementation:

**Grand mean:**
```sql
mu_hat = Σ_i(w_i × X̄_i) / Σ_i(w_i)
```
Weighted average of group means, weighted by exposure.

**EPV (v_hat) — within-group variance:**
```sql
v_hat = [Σ_i Σ_j w_{ij} × (X_{ij} - X̄_i)²] / Σ_i(T_i - 1)
```
Sum of within-group squared deviations, weighted by exposure, divided by the total number of within-group degrees of freedom. Groups with only one period (T_i = 1) contribute zero to the numerator but would subtract 1 from the denominator — filter these out before computing v_hat.

**VHM (a_hat) — between-group variance:**
```sql
c    = Σ_i(w_i) - Σ_i(w_i²) / Σ_i(w_i)
s²   = Σ_i w_i × (X̄_i - mu_hat)²
a_hat = (s² - (r - 1) × v_hat) / c
```
The between-group sum of squares, with sampling noise removed. Important: a_hat can be negative. This happens when within-group variance dominates — the groups look similar not because they are truly similar, but because the data are too noisy to distinguish them. By convention, truncate a_hat at zero. When a_hat = 0, K = infinity and all Z_i = 0 — every group gets the portfolio mean. This is not wrong; it means the data cannot justify any group-level adjustment.

### The implementation

Create a new cell with:

```python
%md
## Part 4: Bühlmann-Straub implementation
```

Create the next cell with the full function:

```python
def buhlmann_straub(
    data: pl.DataFrame,
    group_col: str,
    value_col: str,
    weight_col: str,
    log_transform: bool = True,
) -> dict:
    """
    Bühlmann-Straub credibility estimator.

    Parameters
    ----------
    data : Polars DataFrame with one row per (group, period).
        Each row represents one group in one period.
    group_col : str
        Column identifying the group (e.g. "postcode_district").
    value_col : str
        Observed loss rate per unit of exposure (e.g. "claim_frequency").
    weight_col : str
        Exposure weight (e.g. "earned_years").
    log_transform : bool
        If True, apply B-S in log-rate space to avoid the Jensen's inequality
        bias that arises in multiplicative (log-link) frameworks. Set False for
        additive models or when working with severity in additive space.

    Returns
    -------
    dict with keys:
        grand_mean  : float — portfolio mean (on original scale if log_transform=True)
        v_hat       : float — EPV estimate (on working scale)
        a_hat       : float — VHM estimate, truncated at 0
        a_hat_raw   : float — VHM before truncation (negative values are diagnostic)
        k           : float — v_hat / a_hat (inf if a_hat = 0)
        results     : Polars DataFrame with per-group Z and credibility estimates
    """
    # --- Work in log space if multiplicative framework ---
    if log_transform:
        # Clip near-zero rates to avoid log(0). Do not clip to 0 directly:
        # that silently drops valid thin-cell observations.
        data = data.with_columns(
            pl.col(value_col).clip(lower_bound=1e-9).log().alias("_y")
        )
        y_col = "_y"
    else:
        y_col = value_col

    groups = data[group_col].unique().sort().to_list()
    r = len(groups)

    # --- Per-group sufficient statistics ---
    group_data = (
        data
        .group_by(group_col)
        .agg([
            pl.col(weight_col).sum().alias("w_i"),
            (
                (pl.col(y_col) * pl.col(weight_col)).sum()
                / pl.col(weight_col).sum()
            ).alias("x_bar_i"),
            pl.col(weight_col).count().alias("T_i"),   # number of periods
        ])
        .sort(group_col)
    )

    # Filter single-period groups for EPV calculation.
    # A group with T_i = 1 contributes 0 to the EPV numerator (no within-group
    # deviation possible from a single observation) but would subtract 1 from
    # the denominator. This incorrectly reduces the effective sample size.
    group_data_epv = group_data.filter(pl.col("T_i") > 1)

    w_i = group_data["w_i"].to_numpy()
    x_bar_i = group_data["x_bar_i"].to_numpy()
    w = w_i.sum()

    # --- Collective mean (grand mean) ---
    mu_hat = (w_i * x_bar_i).sum() / w

    # --- EPV: within-group variance ---
    def epv_numerator_for_group(grp_name: str) -> float:
        grp = data.filter(pl.col(group_col) == grp_name)
        if grp.height <= 1:
            return 0.0
        x_bar = float(
            (grp[y_col] * grp[weight_col]).sum() / grp[weight_col].sum()
        )
        resid_sq = (grp[y_col].to_numpy() - x_bar) ** 2
        return float((resid_sq * grp[weight_col].to_numpy()).sum())

    epv_groups = group_data_epv[group_col].to_list()
    epv_num = sum(epv_numerator_for_group(g) for g in epv_groups)
    epv_den = float(group_data_epv["T_i"].sum() - len(epv_groups))
    v_hat = epv_num / epv_den if epv_den > 0 else 0.0

    # --- VHM: between-group variance ---
    c = w - (w_i ** 2).sum() / w
    s_sq = (w_i * (x_bar_i - mu_hat) ** 2).sum()
    a_hat_raw = (s_sq - (r - 1) * v_hat) / c
    # Truncate at zero: negative a_hat means data cannot distinguish groups.
    # This is a valid conclusion, not an error. When a_hat = 0, all Z_i = 0.
    a_hat = max(a_hat_raw, 0.0)

    # --- Credibility factors and estimates ---
    k = v_hat / a_hat if a_hat > 0 else np.inf
    z_i = w_i / (w_i + k) if np.isfinite(k) else np.zeros(r)
    cred_est_log = z_i * x_bar_i + (1 - z_i) * mu_hat

    # Convert back from log space if needed
    if log_transform:
        grand_mean = np.exp(mu_hat)
        cred_est = np.exp(cred_est_log)
        obs_mean = np.exp(x_bar_i)
    else:
        grand_mean = mu_hat
        cred_est = cred_est_log
        obs_mean = x_bar_i

    results_df = pl.DataFrame({
        group_col: group_data[group_col].to_list(),
        "exposure":               w_i.tolist(),
        "obs_mean":               obs_mean.tolist(),
        "Z":                      z_i.tolist(),
        "credibility_estimate":   cred_est.tolist(),
    })

    return {
        "grand_mean":  grand_mean,
        "v_hat":       v_hat,
        "a_hat":       a_hat,
        "a_hat_raw":   a_hat_raw,
        "k":           k,
        "results":     results_df,
    }
```

**What this does:** Defines the `buhlmann_straub` function. It takes a Polars DataFrame with one row per (group, period) — in our case, one row per (postcode_district, accident_year) — and returns the structural parameters and per-group credibility estimates. The `log_transform=True` argument tells it to work in log-rate space, which is correct for Poisson frequency models.

**Run this cell.** There is no output — you are just defining the function.

### Fitting Bühlmann-Straub on claim frequency

Now apply it. First, prepare the district-year data in the format the function expects:

```python
# The function needs one row per (group, period) with:
#   - the loss rate (claim frequency for that district-year)
#   - the exposure weight (earned years for that district-year)

# Filter out district-years with very low exposure to avoid near-infinite
# frequencies from near-zero denominators.
# 0.5 years is the minimum threshold — anything below this is a data artefact
# (mid-year new entrant or incomplete year) rather than a real observation.
dist_year = (
    df
    .filter(pl.col("earned_years") > 0.5)
    .select(["postcode_district", "accident_year", "earned_years",
             "claim_count", "claim_frequency"])
)

print(f"Rows in dist_year: {dist_year.height}  (should be close to 600)")
print(f"\nPreview:")
print(dist_year.head(8))
```

**Run this cell.** You should see approximately 600 rows (600 = 120 districts × 5 years; a few near-zero exposure rows may be filtered out).

Now fit:

```python
bs = buhlmann_straub(
    data=dist_year,
    group_col="postcode_district",
    value_col="claim_frequency",
    weight_col="earned_years",
    log_transform=True,    # working in log-rate space: correct for Poisson/log-link
)

print("=== Bühlmann-Straub Results ===")
print()
print(f"Portfolio grand mean:  {bs['grand_mean']:.4f}  ({bs['grand_mean']*100:.2f}% frequency)")
print()
print(f"EPV (v):   {bs['v_hat']:.6f}   within-district year-to-year variance")
print(f"VHM (a):   {bs['a_hat']:.6f}   between-district variance (true underlying differences)")
print(f"K:         {bs['k']:.1f}         earned years for Z = 0.50")
print()

if bs['a_hat_raw'] < 0:
    print(f"DIAGNOSTIC: a_hat before truncation = {bs['a_hat_raw']:.6f}  (negative)")
    print("  The data cannot distinguish district effects from sampling noise.")
    print("  All Z = 0; every district gets the portfolio mean.")
    print("  This is unusual for a 120-district synthetic dataset — check data quality.")
else:
    print(f"  a_hat raw (before truncation): {bs['a_hat_raw']:.6f}  (positive — good)")

print()
print("Per-district credibility estimates (first 15):")
print(bs['results'].head(15))
```

**What this does:** Runs the Bühlmann-Straub estimator on all 120 districts across 5 years. The output tells you: the portfolio mean frequency, the within-district variance (EPV), the between-district variance (VHM), and the K parameter that controls the blend.

**Run this cell.**

**What you should see:**
- Grand mean close to 7% (our true portfolio frequency)
- VHM (a) positive — the simulation has genuine between-district heterogeneity (sigma=0.35)
- K somewhere in the range 100-800 (depending on the simulated data)
- A DataFrame with 120 rows, one per district, showing Z values ranging from near 0 (thin districts) to near 1 (dense districts)

If a_hat is negative, something has gone wrong with the data. Re-run the data generation cell with `rng = np.random.default_rng(seed=42)` and try again.

### What K means in practice

```python
# Exposure thresholds for different Z values, given this K
k = bs['k']
print(f"With K = {k:.0f} earned years:")
print()
print(f"  Z = 0.25  →  need {k/3:.0f} earned years (75% from portfolio mean)")
print(f"  Z = 0.50  →  need {k:.0f} earned years  (half weight on own experience)")
print(f"  Z = 0.67  →  need {2*k:.0f} earned years  (two-thirds on own experience)")
print(f"  Z = 0.80  →  need {4*k:.0f} earned years  (80% on own experience)")
print(f"  Z = 0.90  →  need {9*k:.0f} earned years  (90% on own experience)")
print()
print("In context: a district needs K earned years for Z = 0.50.")
print(f"A thin district with 100 earned years has Z = {100/(100+k):.2f}")
print(f"A dense district with 5,000 earned years has Z = {5000/(5000+k):.2f}")
```

**Run this cell.** The output shows how many policy-years a district needs before its own experience dominates the portfolio mean. This is the number you bring to the underwriting team when explaining why thin district rates are constrained.

### Checkpoint 2: Validate credibility estimates against the true rates

Since we generated the data, we can check whether the credibility estimator is actually recovering the true rates better than the naive observed rates:

```python
# Merge credibility estimates with true rates for validation
bs_results = bs["results"]

# Get true rates from the aggregated district totals
true_rates_df = dist_totals.select(["postcode_district", "true_rate", "total_earned_years"])

validation = (
    bs_results
    .join(true_rates_df, on="postcode_district", how="inner")
    .with_columns([
        # Error of naive observed rate vs true rate
        ((pl.col("obs_mean") - pl.col("true_rate")).abs()).alias("obs_error"),
        # Error of credibility estimate vs true rate
        ((pl.col("credibility_estimate") - pl.col("true_rate")).abs()).alias("cred_error"),
    ])
)

# Mean absolute error comparison
mae_obs = float(validation["obs_error"].mean())
mae_cred = float(validation["cred_error"].mean())

print("=== CHECKPOINT 2: CREDIBILITY VALIDATION ===")
print()
print("Comparing naive observed rates to credibility estimates vs the true rate:")
print(f"  MAE (observed rate):           {mae_obs:.5f}  ({mae_obs*100:.3f}%)")
print(f"  MAE (credibility estimate):    {mae_cred:.5f}  ({mae_cred*100:.3f}%)")
print()
improvement = (mae_obs - mae_cred) / mae_obs * 100
print(f"  Credibility reduction in MAE:  {improvement:.1f}%")
print()

if improvement > 0:
    print(f"Credibility estimates are closer to the truth on average. Good.")
else:
    print("Observed rates are closer on average. This can happen with a homogeneous")
    print("portfolio (small sigma_district) — check your simulation parameters.")

# Show the districts where credibility helps most (the thinnest ones)
thin = validation.sort("total_earned_years").head(20)
print()
print("20 thinnest districts — credibility vs observed error:")
print(thin.select(["postcode_district", "total_earned_years", "true_rate",
                    "obs_mean", "credibility_estimate",
                    "obs_error", "cred_error"]))
```

**What you should see:** Credibility estimates with lower MAE than naive observed rates, particularly for thin districts. The improvement should be 20-50% for a portfolio with 120 districts and our simulation parameters. For the 20 thinnest districts, the credibility estimate should be substantially closer to the true rate — this is the point of the exercise.