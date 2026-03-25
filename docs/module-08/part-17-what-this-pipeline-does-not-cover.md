## Part 17: What this pipeline does not cover

Every pipeline has boundaries. Here are the known gaps in what we have built and what fills them.

### Production monitoring

The pipeline validates the model on historical held-out data before deployment. It does not include the monitoring framework that detects when the model drifts from actual outcomes in production.

A deployed frequency model should have monthly actual-versus-expected monitoring: actual claim counts by prediction decile versus the model's predictions. When the A/E ratio in the top decile moves from 1.02 to 1.18, the model is underpricing high-risk policies. This is a distinct problem from the initial calibration check in Stage 8.5 — calibration testing is a pre-deployment gate; A/E monitoring is a post-deployment signal.

`insurance_monitoring` provides the tools for production monitoring. The `ae_ratio` function computes the A/E ratio with Poisson confidence intervals. `GiniDriftBootstrapTest` detects when the model's discrimination (rank ordering) has deteriorated. `PITMonitor` provides anytime-valid calibration change detection with formal type I error control. Module 11 covers how to connect these to the pipeline output.

### Multi-peril modelling

The pipeline models motor claims as a single peril. UK private motor combines at least three: own damage (OD), third-party property damage (TPPD), and third-party personal injury (TPPI). Each peril has different frequency, severity, development characteristics, and regulatory treatment.

A single severity model averages across these distributions. A portfolio with an unusual BI severity year will look like ordinary OD volatility in the aggregate model. The rating signal is diluted and the pricing committee cannot distinguish between the perils.

Production pipelines for UK motor should fit separate frequency and severity models per peril, calibrate separate conformal intervals, and combine them for the technical premium. The pipeline architecture in this module supports this: run Stage 6 three times, once per peril, with different target columns. The rest of the pipeline is unchanged.

### Postcode-level geography

Stage 3 uses a six-category flat region factor. UK motor pricing in practice operates at postcode district level (2,300+ districts). A flat region factor misses most of the genuine geographic variation in claim frequency and severity.

Postcode district-level rating requires Bühlmann-Straub or Bayesian hierarchical blending to handle thin cells — a district with 15 policies cannot support a credible relativity estimate without borrowing from neighbours. Module 12 covers spatial territory factor estimation.

One important note for production: postcode district is a proxy discrimination risk under the FCA's Consumer Duty. Any territory factor must be documented with actuarial justification, and its performance across protected characteristic groups (ethnicity, disability) must be assessed before deployment.

### Demand model calibration

The demand model in Stage 9 uses fixed parameters (intercept 1.2, price coefficient -2.0). These are not estimated from data; they are assumptions. If actual renewal elasticity in your book is -0.8 (much less price-sensitive), the optimiser will be overly conservative about rate increases and you will under-recover on loss ratio. If actual elasticity is -3.5 (much more price-sensitive), the optimiser will be too aggressive and you will lose more volume than projected.

Module 9 covers causal demand estimation from observational renewal data using double machine learning. The estimated elasticity should be re-estimated at least annually and inserted into Stage 9's PortfolioOptimiser call.

### Expense loading

The pipeline's technical premium is the expected loss cost only. It does not include:

- Acquisition costs (PCW fees, broker commissions)
- Operating expenses (claims handling, policy administration)
- Investment income offset
- Reinsurance costs

Commercial premiums add loadings for these items. For a portfolio with a PCW acquisition cost of £35 per policy and an operating expense ratio of 12%, the loaded premium is `pure_premium / (1 - 0.12) + 35`. These loadings are applied outside the model, typically at the product or channel level. The pipeline produces the loss cost component only; the actuarial team adds the loadings in the rating engine.

### Claims development loading

The severity model trains on incurred-to-date values. For lines with slow-developing claims (BI, employers' liability), incurred values in recent accident years understate ultimate. The pipeline does not apply development factors before training.

For motor OD with a 6-month IBNR buffer, this is acceptable — claims develop quickly. For motor BI or commercial lines, apply accident year development factors (chain-ladder or Bornhuetter-Ferguson) to each accident year's incurred values before feeding them to Stage 6. The development factors are standard actuarial outputs; they just need to be applied in Stage 2 before the features table is written.
