## Part 1: The problem you are solving

### The postcode district scenario

You are pricing a UK motor book. Your data team has given you claims experience by postcode district for the last five accident years. You have 2,300 postcode districts in the data. You are reviewing the rating for KT (Kingston upon Thames).

KT has 847 policy-years of exposure and 11 claims over five years. The observed claim frequency is:

```sql
11 / 847 = 1.30%
```

The portfolio mean for risks with similar rating factor profiles is 6.8%.

Should KT's rating factor be based on 1.30%? On 6.8%? Something in between?

The answer to "something in between" is not arbitrary. It depends on two things:

1. **How variable is a district's true risk from year to year?** If KT's true risk fluctuates a lot year to year (high within-district variance), then five years of data with 11 claims is not very informative — a run of good luck is plausible. We should put less weight on KT's own experience.

2. **How different are districts from each other genuinely?** If UK postcode districts have genuinely different risk profiles (high between-district variance), then KT probably is different from the portfolio mean, even if we cannot pin down exactly how different. We should put more weight on KT's own experience.

Bühlmann-Straub credibility gives you the mathematically principled formula that answers this question. Bayesian hierarchical models give you the same answer with richer uncertainty quantification. This module covers both.

### Why GLMs and GBMs do not solve it

You might wonder: doesn't the area factor in your GLM handle geographic risk? Partly. But:

A main-effects GLM assigns a single area coefficient to all districts in a postcode area (SW, KT, EC, etc.). Districts within an area share the coefficient — there is no district-level differentiation. If you fit a district-level factor in a GLM, KT gets a coefficient estimated from 847 policies and 11 claims. The standard error on that coefficient is enormous. The GLM will either return an implausible rate or, with regularisation, shrink KT's coefficient to zero — which is also wrong.

Ridge regularisation shrinks every coefficient toward zero, regardless of the cell's exposure. A district with 50,000 policy-years gets the same shrinkage formula as a district with 50. That is not right: the dense district deserves its own experience; the thin district deserves pooling.

LASSO is worse. It forces thin cells to exactly zero, which means the district gets the portfolio base rate. That is not credibility — it is arbitrary censorship of real geographic variation.

GBMs have `min_data_in_leaf`. Set it too low: the model learns noise in KT as signal. Set it too high: KT vanishes into the portfolio average. Neither is principled.

**Credibility theory answers the question directly.** Each cell gets a blend of its own experience and the portfolio mean, in proportion to how much information its experience actually contains. The blend is not a tuning parameter — it is derived from the data's evidence about the heterogeneity of the portfolio.

### Scale of the problem

Great Britain has approximately 2,800 postcode districts. A UK motor book with 1.5 million policies averages around 535 policies per district. That sounds comfortable, but motor books are not uniformly distributed:

- Inner London districts (SW1, EC1, W1) each contain several thousand policies
- Rural Scottish districts (KW, IV, HS) may have under 50 policies
- Thin cells are a structural feature of personal lines rating, not an edge case

Even with 535 policies average, the variance is enormous. Any district with fewer than 200 policy-years of experience — roughly 40% of UK districts in a typical mid-size motor book — needs credibility treatment.