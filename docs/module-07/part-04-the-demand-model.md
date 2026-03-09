## Part 4: The demand model

### Why you need one

The LR constraint and the volume constraint both depend on how many customers renew at the new rates. Without a demand model, you have to assume either:

- Everyone renews regardless of price (unrealistic: volume does not change)
- Lapse rates are fixed regardless of price (unrealistic: rates do not affect who stays)

Neither is right. A demand model tells the optimiser: if you raise this customer's premium by 5%, the probability they renew changes from, say, 68% to 65%. The optimiser accounts for this when computing expected LR and expected volume.

Without a demand model, the volume constraint is not meaningful (it is always satisfied unless you have a separate assumption about lapses), and the LR constraint is overoptimistic (it ignores the fact that rate increases cause lapses, which change the book composition).

### The logistic demand model

The `rate-optimiser` library uses a logistic demand model. This is the workhorse specification for renewal probability in UK personal lines. The renewal probability for policy i at a price ratio p\_i (new premium divided by market premium) is:

```
renewal_prob_i = sigmoid(intercept + price_coef * log(p_i) + tenure_coef * tenure_i)
```

where `sigmoid(x) = 1 / (1 + exp(-x))` is the logistic function. The inputs are:

- `p_i = new_premium_i / market_premium_i`: how expensive this policy is relative to what the customer could get elsewhere
- `log(p_i)`: the log of the price ratio. Using log makes the model multiplicative: a 10% increase from 100% to 110% of market has the same demand effect as a 10% increase from 90% to 99% of market
- `tenure_i`: years the customer has been with the insurer. Longer-tenured customers are stickier — they are less price-sensitive

The key parameter is `price_coef`. It is negative (higher price, lower renewal probability) and is called the **log-price semi-elasticity**. A value of -2.0 means: a 1% increase in log price above market reduces the log-odds of renewal by 2 percentage points.

To understand what that means in practice: if a customer currently has a 60% renewal probability (logit of about 0.41), and we raise their price by 1% above market, the new logit is 0.41 + (-2.0 x 0.01) = 0.41 - 0.02 = 0.39, giving a new renewal probability of sigmoid(0.39) = 59.6%. That is a 0.4 percentage point reduction in renewal probability for a 1% price increase.

For UK motor, the relevant benchmarks from market research and published lapse analyses (e.g., Bain & Company UK motor loyalty studies 2018-2022) are:

- **PCW (price comparison website) channel**: price semi-elasticity typically -1.5 to -3.0. PCW customers have already demonstrated they will shop around. They are the most price-sensitive segment.
- **Direct channel**: -0.5 to -1.5. Direct customers have already chosen not to use a PCW. A modest rate increase is less likely to trigger a lapse.

These are starting points. You must calibrate the demand model against your own observed lapse data before using it in the optimiser.

### What miscalibration looks like

If you use a PCW elasticity of -2.5 when your actual elasticity is -1.2, the optimiser will believe you have far less pricing power than you do. It will think that even a small rate increase causes a large volume loss, and it will constrain the rate action more than necessary. The frontier will show infeasibility at targets that are actually achievable.

If you use -0.8 when your actual elasticity is -2.0, the optimiser will overestimate pricing power. It will produce a frontier that claims 72% LR is achievable at 97% volume retention, but in practice the actual lapses will be much higher and the achieved LR will be worse than the model predicted.

**The demand model must be calibrated before you run the optimiser.** Exercise 1 includes a calibration check. In Part 11 we address the limitations of the logistic specification.