## Part 11: Application 1 - Flagging uncertain risks for underwriting referral

The conformal interval gives you two dimensions for every risk:

1. **Risk level** - the point estimate (£342 is an expensive risk; £87 is a cheap one)
2. **Model uncertainty** - the relative interval width ((upper - lower) / point estimate)

These are independent. A risk can be expensive and well-understood (young driver with conviction points, but there are thousands in training data). A risk can be cheap and poorly understood (unusual feature combination that appears rarely in training).

Underwriting referral should be based on model uncertainty, not risk level. High-risk policies the model understands can be quoted automatically with high confidence. Uncertain policies - regardless of price - should go to a human.

In a new cell:

```python
%md
## Part 11: Underwriting referral flag
```

```python
# Extract arrays for computation
point  = intervals_90["point"].to_numpy()
lower  = intervals_90["lower"].to_numpy()
upper  = intervals_90["upper"].to_numpy()

# Relative width: how wide is the interval relative to the point estimate?
# This measures model uncertainty, not risk level
rel_width = (upper - lower) / np.clip(point, 1e-6, None)

# Set threshold at the 90th percentile -> exactly 10% referral rate
width_threshold = np.quantile(rel_width, 0.90)
flag_for_review = rel_width > width_threshold

print(f"Relative width threshold (90th percentile): {width_threshold:.4f}")
print(f"Policies flagged for review: {flag_for_review.sum():,} ({100*flag_for_review.mean():.1f}%)")
```

**What you should see:**

```python
Relative width threshold (90th percentile): x.xxxx
Policies flagged for review: 2,xxx (10.0%)
```

The flag rate is exactly 10% by construction - you set the threshold at the 90th percentile of relative widths, so the top 10% are flagged. If the underwriting director wants a 5% referral rate, use the 95th percentile. If they want 15%, use the 85th percentile.

Now characterise the flagged risks:

```python
# Build a combined analysis frame
X_test_pl = pl.from_pandas(X_test.reset_index(drop=True))
X_test_pl = X_test_pl.with_columns([
    pl.Series("flagged",    flag_for_review),
    pl.Series("rel_width",  rel_width),
    pl.Series("point_est",  point),
    pl.Series("actual",     y_test.values),
])

# Profile: flagged vs unflagged
for flag_val, label in [(True, "FLAGGED (uncertain)"), (False, "Not flagged")]:
    sub = X_test_pl.filter(pl.col("flagged") == flag_val)
    print(f"\n{label}: {len(sub):,} policies")
    print(f"  Mean point estimate:    £{sub['point_est'].mean():.2f}")
    print(f"  Mean actual incurred:   £{sub['actual'].mean():.2f}")
    print(f"  Mean driver age:        {sub['driver_age'].mean():.1f}")
    print(f"  Mean vehicle group:     {sub['vehicle_group'].mean():.1f}")
    print(f"  % with convictions:     {(sub['conviction_points'] > 0).mean() * 100:.1f}%")
    print(f"  Mean relative width:    {sub['rel_width'].mean():.3f}")
```

**What you should see:** flagged risks skew towards younger drivers, higher vehicle groups, and conviction points. They are generally more expensive than unflagged risks. But the key point - which you need to explain to the underwriting director - is that flagging is based on training data density, not just risk level.

The conversation you should be prepared to have:

"Why are we flagging young drivers with conviction points? Surely we know they are high risk."

Your answer: "Yes, they are high risk and we know that well. We have thousands of such drivers in training data. But we have very few 19-year-olds with 9 conviction points in vehicle group 47. The model's prediction for that specific combination is uncertain, not the prediction for young drivers in general. We are flagging the thin-cell combinations where we genuinely lack data, not the common high-risk profiles where the model is confident."

This distinction matters for Consumer Duty. Referring risks for human review because the **model is uncertain** is a different and more defensible process than discretionary referrals based on underwriter judgment.

```python
# Verify: coverage is similar for flagged and unflagged groups
# (the flag is based on width, not on coverage failure)
for flag_val, label in [(True, "Flagged"), (False, "Not flagged")]:
    mask    = flag_for_review == flag_val
    covered = ((y_test.values[mask] >= lower[mask]) & (y_test.values[mask] <= upper[mask]))
    print(f"{label} actual coverage: {covered.mean():.3f}")
```

Both groups should show coverage close to 90%. If the flagged group has materially lower coverage (e.g. below 85%), it suggests the `pearson_weighted` score is not fully correcting for the heteroscedasticity in those thin cells.