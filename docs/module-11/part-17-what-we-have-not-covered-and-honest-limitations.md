## Part 17: What we have not covered, and honest limitations

### Limitations of the MBBEFD family

**The constant c assumption**: when you use a Swiss Re standard curve, you are assuming the same c-parameter applies to every risk in the portfolio regardless of construction, occupancy, or location. For a mixed portfolio, this is an approximation. The fitted curve approach is better -- it produces an average c -- but the within-portfolio heterogeneity in c is unmodelled.

**The total loss probability**: the MBBEFD point mass at z = 1 is 1/g, which is a single number. In practice, the probability of a total loss depends strongly on construction type, sprinkler protection, location, and building age. A modern steel-framed warehouse with good sprinkler coverage has a very different total loss probability from a Victorian mill without sprinklers. Using a single g-parameter ignores this heterogeneity. For heterogeneous portfolios, consider fitting separate curves by construction class before pooling.

**Catastrophe losses**: the MBBEFD framework is designed for attritional and large individual losses. It does not model catastrophe events, which can affect many risks simultaneously and may produce destruction rates that look different from the attritional distribution. If your claims data includes a Hurricane Bawbag year (2011, UK) or any significant UK flood event, the catastrophe losses will distort the fitted curve. Remove catastrophe-affected claims before fitting and account for cat separately.

**Data requirements**: the guidance in the `insurance-ilf` README suggests 50+ large losses for a credible fit. Below that, the MLE surface is too flat and the uncertainty in the fitted parameters is too wide to distinguish between Swiss Re curve families. With fewer than 50 losses, stick to a judgment-selected standard curve and document why.

**Extended MBBEFD**: the library implements the basic truncated/censored likelihood following Bernegger (1997), extended to the truncated-censored case following standard survival likelihood arguments. Some recent actuarial literature introduces extended MBBEFD classes that allow more flexible tail behaviour. The extension is not currently in `insurance-ilf`. For portfolios with unusually heavy tails (e.g., risks where a partial loss of 90% of MPL is genuinely as likely as a partial of 30%), the standard MBBEFD may underfit the tail.

### When exposure rating is not the right tool

Exposure rating is the method of last resort for thin data. It is not superior to burning cost when experience data is sufficient. The hierarchy is:

1. **Own burning cost** (sufficient large losses in the specific layer): use it. It is unbiased and requires no curve assumptions.
2. **Blended burning cost and exposure rating** (moderate data): use a credibility weighting. The credibility module covers this.
3. **Pure exposure rating** (insufficient data): use the MBBEFD framework from this module.

The threshold for "sufficient" is typically 20-30 large losses in the layer over the experience period. With fewer, the burning cost estimate has too wide a confidence interval to be reliable, and exposure rating provides the necessary stability.

---

This module has covered the theoretical foundations, practical implementation, and honest limitations of MBBEFD exposure curves and ILF tables. The exercises that follow build on this foundation with more complex scenarios.