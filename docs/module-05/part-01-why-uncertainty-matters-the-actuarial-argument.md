## Part 1: Why uncertainty matters - the actuarial argument

Before writing any code, you need to understand why point estimates are insufficient for the three tasks where pricing teams most need uncertainty quantification.

### Minimum premiums

Every commercial minimum premium policy applies some flat uplift: "minimum premium = 1.3x technical premium, subject to a floor of £250." The uplift is chosen based on experience and judgment. It does not vary by risk. A policy with a narrow loss distribution (stable, well-understood risk profile, many similar training observations) gets the same 30% uplift as a policy with a wide loss distribution (unusual risk profile, sparse training data, high inherent volatility).

For the narrow-distribution risk, the 1.3x uplift may be excessive - you are charging a floor well above what the actual upper quantile of losses suggests. For the wide-distribution risk, 1.3x may be dangerously low - the genuine 90th percentile of losses for that risk is far above the technical premium times 1.3.

Both directions create Consumer Duty problems. Overcharging stable risks is a fair value issue. Undercharging volatile risks is a solvency adequacy issue. A risk-specific minimum premium based on the actual calibrated upper bound of losses addresses both simultaneously.

### Reserving

Reserve teams need a range, not just a point estimate. The typical approach - "reserves are 115% of technical premium" - bakes in a fixed margin based on historical reserve adequacy. It does not vary by the composition of the current book. If the current book skews towards more volatile risks (younger drivers, higher vehicle groups, more thin-cell combinations), the fixed percentage understates reserve uncertainty. If it skews towards stable risks, the percentage overstates it.

Conformal prediction intervals aggregate to portfolio-level range estimates that reflect the actual composition of the book. The upper bound of the sum of individual intervals is a genuine 90th-percentile portfolio loss estimate, not a rule of thumb.

### Underwriting referral

The decision about which risks to refer to a human underwriter is often discretionary. A conformal approach makes it systematic: flag the risks where the model's prediction interval is wide relative to the point estimate. These are the risks where the model is genuinely uncertain because the training data is sparse in that region of the feature space. A systematic, data-driven referral process is more defensible to the FCA than an underwriter's discretionary judgment applied to the same risks.

The key conceptual distinction - one that regularly confuses the pricing committee - is between risk level and model uncertainty. A young driver with conviction points is high risk. But if we have 5,000 such drivers in training, the model is not uncertain about that combination. A 72-year-old driving a high-group vehicle with a specific configuration of features might be low-to-moderate risk but appear only a dozen times in training data - the model is very uncertain. Both dimensions exist independently.