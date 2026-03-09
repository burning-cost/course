## Part 3: What is model drift?

### Two types of drift

The word "drift" gets used loosely. It helps to be precise about what can actually change.

**Covariate drift** (also called feature drift or input drift) is when the distribution of your model's input features changes. Your book mix shifts: you start writing more young drivers, or commercial vehicles creep into a portfolio that was historically private. The model's inputs look different from what the model was trained on.

**Concept drift** is when the relationship between inputs and the target changes. Young drivers become relatively safer than they used to be. Urban postcodes that were high-frequency become lower-frequency after a speed camera installation programme. The features have not changed, but what those features *mean* for claim frequency has changed.

These two types of drift require different detection methods and have different implications for action.

Covariate drift can be detected before you have any claims data - just by looking at the incoming risk characteristics. This is what PSI and CSI measure. You know something has changed the moment it changes, not six months later when claims emerge.

Concept drift only becomes visible through outcomes. The A/E ratio is the primary detector. The Gini drift test adds a secondary signal: if the model's discriminatory power has fallen, the feature-risk relationship has weakened.

### Why covariate drift matters even if the model is "right"

Suppose your model was trained and validated on a portfolio where 5% of policies covered drivers under 25. Your current book has grown to 15% under-25 drivers. The model's learned relationship for young drivers is based on a small training sample. At 5% of the book, mispredictions for young drivers barely moved overall metrics. At 15%, they matter a lot.

The model has not changed. The model's performance on young drivers has not changed. But the *importance* of getting young drivers right has tripled. This is enough reason to retrain - not because the model is wrong, but because the model's weak segments now represent a larger share of premium.

PSI catches this. A PSI above 0.2 on driver age, driven by the shift in the under-25 bucket, is a signal that you need to revisit your model's performance specifically for that segment.

### Why concept drift matters even if the book mix is stable

Suppose the book has not changed at all - same age distribution, same regions, same vehicle types. But a new fleet management policy across one of the large hire companies means that commercial-registered vehicles in the database now behave more like private cars. Your model uses vehicle type as a feature. The vehicle type distribution has not changed, but the risk associated with that vehicle type has changed. The model will over-predict for those vehicles.

PSI will not catch this - the feature distribution looks the same. The A/E ratio will catch it, but only after claims emerge. Gini drift may catch it earlier if the model's discrimination has weakened.

### A concrete example: what drift looks like in the data

To make this concrete, here is a stylised example using the motor dataset. In the reference period (2022-2023), claim frequency by driver age band looks like this:

| Driver age | Reference frequency | Reference share |
|------------|--------------------:|----------------:|
| 17-24      | 0.18                | 5%              |
| 25-39      | 0.09                | 35%             |
| 40-59      | 0.07                | 40%             |
| 60+        | 0.08                | 20%             |

In the current period (2024 H1), the book mix has shifted:

| Driver age | Current frequency | Current share |
|------------|------------------:|--------------:|
| 17-24      | 0.19              | 14%           |
| 25-39      | 0.09              | 34%           |
| 40-59      | 0.07              | 37%           |
| 60+        | 0.08              | 15%           |

Two things have happened: the under-25 share has nearly tripled (covariate drift), and the under-25 frequency has nudged up slightly (a small concept drift component).

The portfolio A/E would show:

- Reference average frequency: 0.05*0.18 + 0.35*0.09 + 0.40*0.07 + 0.20*0.08 = 0.082
- If the model predicts reference frequencies for the current mix: 0.14*0.18 + 0.34*0.09 + 0.37*0.07 + 0.15*0.08 = 0.087 (expected)
- Actual current frequency: 0.14*0.19 + 0.34*0.09 + 0.37*0.07 + 0.15*0.08 = 0.088

So A/E = 0.088/0.087 = 1.01. The model looks fine at portfolio level. But the CSI on driver age is elevated (the under-25 share has moved from 5% to 14%), and the under-25 segment A/E is 0.19/0.18 = 1.06. The portfolio-level A/E masked a segment-level issue.

This is exactly why you run CSI and segment A/E, not just the overall A/E.

### The monitoring sequence

We run the metrics in this order:

1. **PSI** on the score distribution - has the distribution of predicted scores changed?
2. **CSI** on each feature - which features have shifted in distribution?
3. **A/E ratio** - are actual claims tracking expected claims?
4. **Gini drift** - has the model's discriminatory power changed?

This sequence matters. If the A/E is fine but PSI is elevated, the book mix has changed but the model is still calibrated - worth watching but not acting. If the A/E is elevated and PSI is elevated, the calibration problem is likely driven by the mix change - recalibration is the appropriate response. If the A/E is elevated but PSI is flat, the underlying risk relationship has changed - this is a concept drift problem that may require retraining rather than just recalibration.

### What monitoring is not

Monitoring is not the same as model validation. Validation (done once, at model deployment) asks "is this model fit for purpose?" Monitoring (done continuously, throughout deployment) asks "is the model still fit for purpose?"

Monitoring metrics are simpler than validation metrics. They are designed to be computed quickly on incoming data, with automated thresholds that flag issues without requiring a human to interpret every output. The human reviews the red flags, not every monthly run.

The `MonitoringReport` class encodes these thresholds. Green, amber, and red lights are assigned automatically. Part 8 shows how to build and read the report. First, we need to understand each metric.
