# Module 7: Constrained Rate Optimisation

In Module 6 you learned how to blend sparse experience with portfolio priors using Bühlmann-Straub credibility and Bayesian hierarchical models. You can now produce reliable factor relativities even for thin cells. This module is about what you do with those relativities next: deciding how much to move the rates, which factors to move, and how to prove to the FCA that the decision was principled.

Every pricing actuary has a version of this problem. The book is running above target. The commercial director wants as little volume disruption as possible. The underwriting committee has set factor movement caps. The FCA expects ENBP compliance for every renewal policy. There are four things you need to satisfy simultaneously, and the spreadsheet approach treats them sequentially, by judgment, in a way that is impossible to audit.

This module replaces that process with a formally stated optimisation problem. The solution is a vector of factor adjustments that minimises customer disruption subject to simultaneously satisfying the loss ratio target, the volume floor, the FCA's fair pricing rules, and the underwriting movement caps. You can solve it in seconds, plot the full trade-off for the pricing committee, and export the result to a Delta table with a complete audit trail.

By the end of this module you will have:

- Understood the constrained optimisation problem in plain English before touching any maths
- Built a demand model that knows how renewal probability responds to price changes
- Set up and solved the four-constraint rate optimisation problem using SLSQP
- Interpreted shadow prices and understood what they tell a commercial director
- Traced the efficient frontier and identified the pricing committee's decision point
- Verified ENBP compliance per-policy, not just at the aggregate level
- Fixed the cross-subsidy analysis to show what it actually shows
- Extended the optimiser to handle stochastic loss ratio targets using chance constraints
- Documented the limitations of this approach honestly
[Download the notebook for this module](notebook.py)

---