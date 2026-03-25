## Part 16: What we have not covered

### The autodml subpackage

`insurance_causal` has a separate `autodml` subpackage (`PremiumElasticity`, `DoseResponseCurve`) implementing automatic debiased ML via minimax Riesz regression (Chernozhukov et al. 2022, *Econometrica* 90(3): 967–1027). This avoids the need to explicitly estimate E[D|X] — useful when the treatment distribution is difficult to model.

For binary outcomes and log price change treatments, the `elasticity` subpackage (which we have used throughout) is the right choice. `autodml` is more appropriate when the treatment is non-standard or when you want to estimate dose-response curves over a continuous treatment range rather than a single semi-elasticity.

### The causal_forest subpackage

The `causal_forest` subpackage provides formal heterogeneous treatment effect inference: BLP tests (does heterogeneity exist?), GATES by quantile group, CLAN (characterisation of the least and most affected subgroups), and RATE/AUTOC targeting evaluation.

We used `RenewalElasticityEstimator` from the `elasticity` subpackage, which wraps the same `CausalForestDML` estimator. The `causal_forest` subpackage provides `HeterogeneousElasticityEstimator` with additional formal testing. If you need a p-value for "does price elasticity vary significantly by NCD band?" rather than just a GATE table, use `HeterogeneousInference.run()`.

### Instrumental variables

When the near-deterministic price problem is present and A/B testing is not available, a valid instrument — a variable that affects price but is independent of renewal probability conditional on risk factors — allows consistent estimation via PLIV (partial linear IV).

Practical instruments:

- **Bulk rate change indicator**: all policies subject to a 10% Q1 2024 bulk increase share exogenous price variation across otherwise similar customers
- **PS21/5 kink**: customers previously above ENBP received forced price reductions in January 2022 — discontinuity in price change at the ENBP boundary provides quasi-experimental variation (with caveats about selection at the boundary)
- **Competitor withdrawal**: a major competitor leaving a segment creates price shocks on aggregators that are plausibly exogenous at the individual level

### Survival models for CLV pricing

The renewal probability from the DML model is a one-period estimate. For customer lifetime value pricing — deciding what price maximises expected total profit over the customer's lifetime — you need a survival function: the probability of still being a customer at t = 1, 2, 3, 5 years.

The `causal_forest.exposure` submodule (`prepare_rate_outcome`) handles exposure-weighted outcomes for panel data. Full CLV pricing requires connecting this to a long-run customer model, which is beyond the scope of this module.

### Multi-product demand effects

If you write both motor and home, a motor lapse may trigger a home cancellation. Single-product demand models miss these cross-product effects. They matter for CLV calculations and for understanding the true cost of a lapse. This is currently a research-stage problem — no off-the-shelf solution handles it cleanly.
