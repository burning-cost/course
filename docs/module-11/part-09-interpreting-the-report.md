## Part 9: Interpreting the report

### The five scenarios you will actually encounter

Theory is one thing. In practice, monitoring reports fall into a small number of patterns. Here is how to read each one.

---

**Scenario 1: All green**

PSI < 0.10, A/E within [0.95, 1.05], Gini stable (p-value >= 0.32), all CSI < 0.10.

Action: Log the result, no further investigation. File the report as evidence of ongoing monitoring. This is the outcome for most months on a stable book.

One caution: "all green" in a month with low claim count (e.g., a new book that has only been live for two months) should be treated with more scepticism. The confidence intervals will be wide, and the CI-contains-1.0 test will pass trivially because you do not have enough data to detect even a large A/E shift. Check the expected claim count in the report. If it is below 100, the A/E signal is not yet informative.

---

**Scenario 2: Elevated PSI, green A/E, stable Gini**

The score distribution has shifted (PSI > 0.10), but actual claims are tracking expected (A/E fine) and discrimination is unchanged (Gini stable).

This means the book mix has changed, and the model is correctly applying risk loads to the new mix. The model is doing its job. The score distribution shift is a consequence of the new risks being written, not a failure.

Action: Log the CSI results to understand which features have shifted. No model action required. Note in the monitoring log that the book mix has changed, so future PSI results should be compared against the current distribution rather than the historical one. Consider updating the reference distribution.

---

**Scenario 3: Elevated PSI, elevated A/E, stable Gini**

The book mix has changed (PSI > 0.10), the model is under- or over-predicting overall (A/E outside [0.95, 1.05]), but discrimination is fine (Gini stable).

This is a calibration problem driven by mix shift. The model is correctly ranking risks but at the wrong overall level. The likely explanation: the model was calibrated on a different mix, and the current mix has more (or fewer) high-risk risks than assumed at calibration. The model's overall intercept needs adjusting, but the risk ordering is fine.

Action: Apply a recalibration factor (Part 13 covers this). The factor is simply the portfolio A/E ratio: multiply all predictions by 1/A/E to restore calibration. This is a temporary fix while you investigate whether retraining is needed.

---

**Scenario 4: Green PSI, elevated A/E, falling Gini**

The book mix has not changed materially (PSI fine), but claims are coming in higher than expected (A/E > 1.05) and the model is less able to discriminate (Gini falling).

This is concept drift. The relationship between the features and the outcome has changed. The model was trained on one risk environment and deployed in a different one. The features still describe the same kinds of risk, but those risks are now behaving differently.

This pattern often appears after a regulatory change or a macroeconomic shock - the whiplash reform example from Part 1 is a textbook case of this. The model's features did not change, but what they meant for severity changed discontinuously at the reform date.

Action: This is the most serious pattern. A recalibration factor will fix the average but not the discrimination. The portfolio may look correctly priced in aggregate but incorrect at the policy level - dangerous for competitive positioning and for Consumer Duty. Escalate to head of pricing. Begin retraining work. Apply a temporary recalibration factor as a holding measure.

---

**Scenario 5: Elevated CSI on one feature, all other metrics green**

A single feature has shifted (CSI > 0.20) but the portfolio A/E is fine, Gini is stable, and score PSI is low.

This means the feature has shifted but it is not a significant pricing driver. If vehicle group CSI spikes because a new van insurance product is being written, but the model's treatment of vehicle group is not a major driver of predictions, the portfolio impact is small.

Action: Investigate the feature shift. Is it a genuine mix change or a data quality issue upstream? If genuine, assess whether the model's learned relationship for that feature remains appropriate. Flag for review at the next model validation cycle. No immediate model action required.

---

### Reading the CSI table in combination with A/E

When A/E is elevated, use the CSI table to find the candidate features. Look for features with CSI > 0.10 and ask: if this feature's distribution has shifted in this direction, would you expect the A/E to move in the direction it has?

Example: if the young driver proportion has grown (CSI elevated on `driver_age`, contribution concentrated in the under-25 bin) and A/E > 1.0, the question is: are young drivers in the current data claiming more than the model expected? Compute A/E for the under-25 segment specifically. If A/E for under-25s is 1.30 and the overall A/E is 1.08, the elevated overall A/E is almost entirely explained by under-25 performance. The fix is model improvement for that segment, not a blunt portfolio recalibration.

### What not to do

Do not over-react to a single month's results. A single month with A/E of 1.08 and PSI of 0.12 is not an emergency. One month is not a trend. Before taking model action, you want to see the signal persist for at least two consecutive months, or a single month where one metric is red (not just amber).

Do not under-react to sustained amber. A model that has shown amber A/E for four consecutive months is more concerning than a model that briefly went red and then recovered. Four months of A/E at 1.07 (cumulative 1.07 forecast error over four months) represents a meaningful mis-pricing that compounds.

The monitoring log in Delta (Part 10) gives you the trend view. Never interpret a single month's report in isolation from the trend.
