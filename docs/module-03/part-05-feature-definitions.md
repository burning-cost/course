## Part 5: Feature definitions

Before fitting anything, we need to decide which features go into the model and which are categorical.

Create a new cell:

```python
# Feature definitions for the motor GBM
CONTINUOUS_FEATURES = ["driver_age", "vehicle_group", "ncd_years"]
CAT_FEATURES        = ["area", "conviction_points"]
FEATURES            = CONTINUOUS_FEATURES + CAT_FEATURES

FREQ_TARGET  = "claim_count"
EXPOSURE_COL = "exposure"
SEV_TARGET   = "incurred"
```

Run it. There is no output - you are just setting up variable names that the rest of the notebook will use.

**Why `conviction_points` is categorical:** The values are 0, 3, 6, and 9. These are penalty point totals. They are not continuous quantities: the step from 0 to 3 points (one minor offence) is qualitatively different from 6 to 9 (approaching disqualification territory). Treating them as a continuous number would impose a linear assumption on the effect. As a categorical, CatBoost learns the effect of each penalty level independently using ordered target statistics.

**Why `vehicle_group` is continuous:** ABI groups 1-50 have a roughly monotone relationship with risk - higher groups are generally more expensive vehicles. The continuous treatment allows CatBoost to find non-linear effects within that trend, which tree splits handle naturally.

**Why `driver_age` is continuous:** Age has a well-known non-linear effect - a peak of risk in the under-25 band and a secondary peak in the over-70 band, with a flat middle. Tree splits on a continuous variable capture this shape without requiring manual bucketing.
