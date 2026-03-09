## Part 1: Why the risk model is not enough

### The half the premium everyone ignores

A UK motor pricing review usually goes like this. The claims team builds or refreshes the frequency and severity models. The pure premium model is updated. The technical premium is rebased. The rating factors are checked for drift. That work typically takes weeks.

At the end, the pricing actuary takes the technical premium output and asks: what should we actually charge?

This is where demand enters. The commercial loading - the ratio of the quoted price to the technical premium - is where you make or lose money on volume. If you load too high, customers do not buy, and your quote-to-bind rate collapses. If you load too low, you are profitable on every policy but less profitable than you could have been.

Most teams handle this problem by intuition or by anchoring to the market. They look at what competitors charge and price at or near the market rate. That is a viable strategy but it is not an optimised one. You are leaving money on the table for inelastic customers and you may be over-priced for elastic ones.

The demand model answers the question the risk model cannot: given this price, what is the probability that the customer buys?

### The two problems demand modelling solves

There are two distinct questions worth separating clearly.

**Question 1: Static demand.** At the current quoted price, what is the expected conversion or renewal rate? This is what the conversion model and retention model answer. They are useful for volume forecasting, for monitoring whether your conversion rate is tracking plan, and for identifying segments where your rate is structurally off-market. A well-calibrated conversion model tells you where you are, not what would happen if you moved.

**Question 2: Dynamic demand (elasticity).** If I increase this customer's price by 5%, how much does their probability of buying change? This is what the elasticity model answers. You need this for optimisation: you cannot find the profit-maximising price without knowing how demand responds to price changes.

The industry sometimes conflates these. Akur8 and Earnix both handle conversion modelling. The elasticity question - the truly hard one - is where the methodology gets interesting and where most teams have the weakest foundations.

### What PS21/5 added to the picture

FCA PS21/5 (General Insurance Pricing Practices, effective January 2022) introduced the ENBP rule: a renewing customer must not be quoted more than the equivalent new business price (ENBP). The ENBP is the price a new customer with the same risk profile would be quoted on the same channel today.

This matters for demand modelling in two ways.

First, it adds a hard constraint to the renewal pricing optimisation. You cannot simply charge the profit-maximising renewal price; you can only charge up to the ENBP. If the profit-maximising price for an inelastic long-tenure customer would have been ENBP + 8%, you leave that 8% on the table.

Second, it changes what demand modelling is for in the renewal context. You cannot use the lapse model to justify charging inelastic customers more - that is the loyalty penalty the rule bans. You can use it to identify customers who are at risk of lapsing and offer them a targeted retention discount. These are directionally opposite uses of the same model, and the difference matters to the compliance function.

We come back to ENBP compliance in Part 13. For now, understand that PS21/5 did not make demand modelling less useful; it made it more constrained, and that constraint has a cost we can quantify.

One consequence that is worth naming plainly: post-PS21/5, many UK motor insurers reduced their renewal uplift to near-zero for high-tenure customers to avoid ENBP breaches. For a large fraction of the book, the ENBP constraint now binds -- the FCA has effectively set the price. If your book's average ENBP headroom is less than 2%, the profit optimisation in Part 12 is largely theoretical for those customers. The actionable set is customers with meaningful headroom below the ENBP ceiling. The demand model still tells you who is at lapse risk and should receive a retention discount; it just cannot be used to raise prices on inelastic customers who are already at the ENBP.