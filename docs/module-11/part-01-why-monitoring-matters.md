## Part 1: Why monitoring matters

### You deployed the model. Now what?

When you trained the model in Module 8, you measured its performance on held-out data from the same time period as the training data. That test set Gini told you how well the model discriminates on data that looks like training data. It told you almost nothing about how the model will perform six months from now.

This is not a flaw in the training process. It is the nature of supervised learning applied to a changing world. A model trained on 2022-2023 UK motor data learned the relationship between driver age, vehicle group, geography, and claim frequency as it existed in 2022-2023. That relationship is not fixed.

Consider what has changed in UK motor in recent years:

**COVID-19 lockdowns (2020-2021).** Miles driven dropped sharply. Claim frequency fell. A model trained pre-COVID would have over-predicted frequency during lockdown - A/E ratio below 1.0. A model trained during COVID would under-predict frequency after restrictions lifted - A/E above 1.0.

**Whiplash reform (May 2021).** The Civil Liability Act 2018 changed how soft-tissue injury claims are assessed. Small injury claims moved through the new Official Injury Claim portal. The severity distribution for third-party claims shifted discontinuously at the reform date.

**Ogden rate change (August 2019, then July 2023).** The Ogden rate determines how large catastrophic injury lump sums are calculated. When it changed from -0.75% to -0.25% in 2019, catastrophic injury reserves moved across the industry. A frequency/severity model trained before the change had the wrong severity relationship baked in.

**Used car price inflation (2021-2022).** Average sum insured for comprehensive policies rose significantly as second-hand car values spiked post-COVID. A vehicle age model trained on pre-2021 data saw a different relationship between age and value.

None of these events are predictable. All of them affect model performance. The only way to detect them is to monitor the model continuously against incoming data.

### The minimum you must do

The minimum acceptable monitoring for a deployed pricing model is a monthly **actual vs expected (A/E) ratio** on claim frequency: how many claims actually occurred, divided by how many the model expected. An A/E ratio persistently outside [0.95, 1.05] means the model is systematically wrong in one direction.

Everything else in this module builds on that foundation: ways to diagnose *why* the A/E has drifted, how to detect drift before it shows up in the A/E, and how to decide what action to take.

### Regulatory context: PRA SS1/23

The Prudential Regulation Authority Supervisory Statement SS1/23 ("Model risk management principles for banks") applies to insurers via their internal model governance. While SS1/23 was drafted primarily for banks, the PRA has made clear through Dear CEO letters in 2023 and 2024 that insurers are expected to maintain documented model monitoring frameworks for material pricing models.

The key expectations in SS1/23 relevant to pricing model monitoring are:

- **Ongoing monitoring**: models should be monitored for continued appropriateness throughout their deployment lifecycle
- **Performance metrics**: monitoring should include quantitative metrics appropriate to the model's use
- **Escalation triggers**: thresholds should be defined in advance, with a documented response process for when they are breached
- **Documentation**: monitoring outcomes should be recorded and available for review

We address all four of these in this module. Part 15 covers producing the evidence pack that satisfies a PRA review.

### What "monitoring" actually means day to day

In practice, model monitoring means running a notebook (or automated job) once a month that produces a one-page summary with green, amber, and red lights:

- **Green**: model performing as expected, no action required
- **Amber**: early warning signal, investigate but no immediate action
- **Red**: significant deterioration, escalate to head of pricing, likely requires model action

The `MonitoringReport` class in `insurance-monitoring` generates exactly this summary. The rest of this module shows you how to build it, interpret it, and act on it.
