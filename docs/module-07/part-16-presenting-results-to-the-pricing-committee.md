## Part 16: Presenting results to the pricing committee

This is the part most technical courses skip. The optimiser produces a solution; getting a pricing committee to accept it requires a different skill.

### The structure of the conversation

The pricing committee will ask five questions, approximately in this order:

**"Can we just see the headline number?"**

Yes. The factor adjustments are 3.6-3.7% on each shared factor, zero on the tenure discount. The combined effect is a 15.1% premium increase for the average customer. The expected LR at new rates is 72.0%, and we expect to retain 97.3% of current volume.

**"What if lapses are worse than the model predicts?"**

Show the stochastic result. At 90% confidence (planning to a distribution, not just the mean), the factor adjustments would need to be 4.3-4.5% rather than 3.6-3.7%. The committee can decide whether to price to the expected value or to the 90th percentile.

**"What if we wanted to do less rate?"**

Show the frontier table. Relaxing the LR target from 72% to 73% reduces the factor adjustments to approximately 2.8% and improves expected volume retention to 97.9%. The committee can choose any point on the frontier; you are presenting the full trade-off rather than a single recommendation.

**"Why can we not take more rate on [specific factor]?"**

This is about the factor movement caps. If the underwriting director approved [0.90, 1.15] movement caps, the optimiser cannot exceed them. If the committee wants to take, say, a 20% movement on the vehicle factor, they need to approve a wider mandate and re-run the optimiser. This conversation is healthy: it surfaces the implicit constraint that previously existed only in the underwriting director's judgment.

**"Are we treating any customers unfairly?"**

Show the cross-subsidy analysis. The percentage change is uniform across all customer segments. Young drivers see a larger absolute increase (£91 vs £46) because their base premium is higher, not because the rate action targets them disproportionately. This is the Consumer Duty evidence.

### What to put on the slide

The one-slide summary for the pricing committee:

| Item | Value |
|------|-------|
| LR target | 72.0% |
| Expected LR at new rates | 72.0% |
| Expected volume retention | 97.3% |
| Factor adjustments (shared) | +3.6% to +3.7% |
| Tenure discount adjustment | 0.0% (ENBP constraint) |
| Customer impact (mean) | +15.1% / +£72 per year |
| ENBP compliance | Verified per-policy (0 violations) |
| Solver converged | Yes |

Below the table, the efficient frontier chart. Below that, the constraint binding summary: "LR constraint is binding at the optimum. Volume constraint is not binding (expected volume 97.3% vs 97.0% floor). ENBP constraint is binding (tenure discount cannot move above 1.0). Factor bounds are not binding."