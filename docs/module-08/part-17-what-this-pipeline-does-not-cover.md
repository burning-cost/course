## Part 17: What this pipeline does not cover

This section documents the known limitations of the pipeline as built. Understanding what the pipeline does not do is as important as understanding what it does.

### Geographic credibility

The pipeline treats `region` as a flat five-category factor. UK motor pricing in practice uses postcode district-level rating with 2,300+ distinct districts. A flat five-category region factor misses most of the genuine geographic variation.

Geographic credibility -- blending thin-cell postcode experience with the portfolio mean using a Bühlmann-Straub or Bayesian hierarchical model -- is covered in Module 6. The correct architecture is to replace the flat `region` factor with Module 6's blended district-level relativities, then feed those relativities into the SHAP factor structure.

Geographic credibility is also the most regulatory-sensitive element of UK motor pricing. The FCA's Consumer Duty and its ongoing work on proxy discrimination means that any postcode-level rating factor needs documented actuarial justification. The risk is not just that a postcode is used as a proxy for a protected characteristic -- the risk is that a spatial smoothing method that looks neutral mathematically may still propagate discrimination at the group level. Before deploying any geographic component in production, document: the smoothing method chosen, why it was preferred over alternatives, how it was validated against FCA proxy discrimination guidance, and how it performs for protected characteristic groups.

### Multi-peril combination

The pipeline models motor claims as a single severity distribution. UK private motor actually combines at least three distinct perils: own damage, third-party property damage, and third-party personal injury. Each has different frequency, severity, development characteristics, and regulatory treatment.

A production pipeline would fit separate frequency and severity models for each peril, calibrate separate conformal intervals, and combine them for the technical premium. The current pipeline's single-severity model will produce a reasonable overall pure premium but cannot differentiate between a book skewed towards large BI claims and one skewed towards frequent small OD claims.

### Live demand elasticity

The rate optimiser in Stage 9 uses a logistic demand model with fixed parameters (intercept=1.0, price_coef=-2.0). In production, the demand parameters should be estimated from your own book's renewal lapse experience, ideally re-estimated quarterly. A demand model with incorrect elasticity produces suboptimal rate actions: if you overestimate price sensitivity, the optimiser will be overly conservative with rate increases and you will under-recover on loss ratio.

### Monitoring and drift detection

The pipeline produces a model and validates it on historical data. It does not include a monitoring framework for detecting when that model's predictions drift from actual outcomes in production. Production model monitoring -- actual-versus-expected tracking, SHAP drift monitoring, coverage recalibration triggers -- is a separate pipeline that should run monthly against the production scoring engine.

### Claims development loading

The severity model trains on incurred-to-date values. For accident years that have not fully developed, the incurred values understate ultimate severity. The pipeline does not apply a development loading to the training severities before fitting the model. For motor property damage with a six-month IBNR buffer, this is acceptable -- most OD claims are settled within three months. For motor BI or commercial lines, a chain-ladder development factor should be applied to each accident year's severity before training.

### Expense loading

The technical premium from the pipeline is the expected loss cost only. It does not include acquisition costs, operating expenses, investment income offset, or reinsurance cost. A commercially deployed technical premium adds a loading to the model output for these items. The loading is typically applied at the product/channel level rather than at the individual policy level.

### Reinstatement and policy limits

The severity model predicts mean claim cost without applying per-claim excess structures or policy limits. For policies with material excess or sub-limits, the model's severity prediction is the ground-up cost. The treaty or XL reinsurance attachment point, and any policy excess, need to be applied separately to convert the model's ground-up prediction to a net-of-reinsurance technical premium.

### Reinsurance structure

Related to the above: the rate optimiser optimises on gross of reinsurance. For a book with material reinsurance, the optimiser should work on a net-of-reinsurance basis where the treaty cost is reflected in the technical premium. This requires a reinsurance cost model that is not included here.