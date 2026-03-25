## Part 1: Why the risk model is not enough

### What the technical premium tells you

The claims team builds a frequency model and a severity model. The outputs combine into a technical premium: the price at which, on average, premiums equal expected claims. Every pricing review starts here.

But the technical premium does not tell you what to charge. It tells you the floor — the point below which you are paying claims from capital. Between the floor and whatever the market will bear, there is a range of viable prices. Where you set the price within that range determines how many customers buy and how much profit you make per policy.

Deciding where in that range to price requires knowing something the risk model is blind to: how sensitive customers are to price changes. A 5% increase above technical premium might barely affect renewal probability for a 55-year-old NCD-5 direct customer, but might cause a 25-year-old PCW customer to lapse immediately. The risk model treats both identically. The demand model distinguishes them.

### The commercial loading decision

In practice, the quoted price is some multiple of the technical premium:

```
quoted_price = technical_premium × commercial_loading
```

The commercial loading is where pricing teams make (or leave) money. In UK motor, that loading varies by channel, customer tenure, and competitive position. The decision is usually made by judgment against market data: if the aggregator shows you are 8% above cheapest, and you believe your brand sustains a 4% premium, you need to decide whether the remaining 4% gap loses you meaningful volume.

"Meaningful" requires a model. Without one, you are guessing. With a biased one, you are guessing systematically in the wrong direction. This module fixes the methodology.

### Two distinct demand questions

Two questions that often get conflated:

**Question 1: Static demand.** At the current price, what fraction of customers will renew? This is what a retention model answers. It is useful for volume forecasting and for identifying segments where your rate is so far off-market that lapses are structural rather than price-driven. It tells you where you are.

**Question 2: Elasticity.** If you increase this customer's price by 5%, how much does their renewal probability change? This is what you need for optimisation. You cannot find the profit-maximising price without knowing the slope of the demand curve.

A retention model gives you a point on the demand curve. An elasticity model gives you the curve itself. Most teams have the former. Very few have the latter estimated properly.

### What PS21/5 changed

FCA PS21/5 (General Insurance Pricing Practices, effective January 2022) introduced the equivalent new business price (ENBP) rule: a renewing customer must not be quoted a price higher than the equivalent new business price — what a new customer with the same risk profile would be quoted today on the same channel.

Before PS21/5, UK motor insurers routinely charged long-tenure customers a "loyalty penalty" — price increases above ENBP justified by the customer's observed inelasticity. That practice is now prohibited at the individual policy level, not just on average.

This changes what demand modelling is for in the renewal context. You cannot use a lapse model to justify charging an inelastic customer more. You can use it to identify customers at lapse risk and offer them a targeted retention discount. These are directionally opposite uses of the same model.

A second consequence, worth naming plainly: post-PS21/5, many UK insurers reduced renewal uplifts to near zero for high-tenure customers to stay clear of ENBP breaches. For a large fraction of renewal books, the ENBP constraint now binds — the FCA has effectively set the price. If your book's average ENBP headroom is under 2%, the profit optimisation in Part 12 is largely theoretical for those customers. The actionable set is customers with meaningful headroom below the ENBP ceiling. Estimating elasticity still matters: it tells you which of those customers will leave if you approach the ceiling.
