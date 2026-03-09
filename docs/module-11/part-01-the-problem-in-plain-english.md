## Part 1: The problem in plain English

### What excess-of-loss pricing actually involves

Suppose you are pricing a per-risk XL layer for a commercial property portfolio. The cedant offers a book of 924 risks, ranging from small shops with a sum insured of £250,000 to large industrial facilities insured for £10 million. The layer sits at £1 million excess of £500,000. The cedant wants you to quote a rate on line.

You could use burning cost: take five years of loss data, count how many losses pierced £500,000, sum the recoveries in the £500k-£1.5m band, divide by the subject premium, and add a loading. That works when you have sufficient large losses in the layer. When you do not, you need another method. The experience is too thin.

The exposure rating method says: instead of counting actual losses in the layer, use the known shape of the loss severity distribution to estimate what fraction of ground-up losses falls into the layer. That fraction, applied to the subject premium, gives the expected layer loss. The exposure curve is the object that describes this shape.

More precisely: the exposure curve G(x) tells you, for a risk with maximum possible loss MPL, what proportion of expected loss lies below x × MPL. G(0) = 0. G(1) = 1. The curve is concave and non-decreasing. Every point on the curve gives you the proportion of expected loss you capture by limiting losses at that fraction of MPL.

For a layer from AP (attachment point) to AP + L (layer ceiling), the fraction of expected loss that falls in the layer is G(min(PL, AP+L)/MPL) minus G(min(PL, AP)/MPL), where PL is the policy limit. Multiply by subject premium. That is the expected layer loss.

The critical inputs are:
1. A fitted exposure curve G(x) appropriate for the class of business
2. A risk profile from the cedant (distribution of risks by sum insured)
3. The layer structure (attachment and limit)

We cover all three in this module.

### Why you cannot use the basic frequency-severity approach

In a standard GLM or GBM pricing model, you model frequency and severity separately, multiply, and get expected loss per risk. For ground-up pricing this works well. But for excess-of-loss work it fails for two reasons.

**First, thin data at high severity.** You might have 10,000 risk-years of exposure but only 6 losses that ever pierced £500,000. You cannot fit a credible severity distribution to 6 points. The exposure curve approach borrows structure from the known shape of the MBBEFD family, using the 10,000 ground-up losses to calibrate a curve and then projecting it up into the layer.

**Second, the policy limit problem.** Your claims data is observed subject to policy limits. A risk with a £2 million limit can never produce a loss above £2 million, regardless of the underlying severity distribution. If you naively fit a severity distribution to observed losses, you fit to truncated and censored data without accounting for those constraints. The exposure curve approach handles this explicitly: the curve is fitted in destruction-rate space (loss divided by MPL), which is naturally bounded between 0 and 1 and invariant to the size of the risk.