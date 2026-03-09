## Part 1: The problem in plain English

### What a pricing review actually involves

You run UK motor insurance. The book has 80,000 policies in force. At the end of Q1 you look at the numbers:

- Current loss ratio: 75.2%
- Target loss ratio: 72.0%
- Volume versus plan: -2.8%

The gap is 3.2 percentage points. You need to close it by adjusting rates. But you cannot simply add 3.2pp to every premium: some customers will lapse, which reduces the denominator (premium) and can make the LR worse if you lose the wrong ones. The amount of rate you need to charge is not the same as the gap you need to close.

Your tariff has five rating factors: age, no-claims discount, vehicle group, region, and a tenure discount for renewals. Each factor has a table of relativities — the multipliers applied to the base rate for each level of the factor. The question is: by how much should you scale each factor table?

The spreadsheet approach is to try combinations. Increase age by 4%, NCB by 3%, vehicle by 2%, region by 3%, leave tenure flat. Calculate the expected LR. Volume is now 97.5% of current. ENBP is satisfied (you check the maximum renewal/NB ratio manually for five example policies). The commercial director accepts it. It goes to pricing committee.

This is not a bad outcome. But it has three structural problems.

**Problem 1: You explored a small part of the space.** There are infinitely many combinations of five factor adjustments that might achieve the LR target. You found one by starting near zero and adjusting by judgment. You cannot know whether it is the best one — the one with the smallest customer disruption — because you did not search the space systematically.

**Problem 2: You cannot quantify trade-offs.** The commercial director asks: "What if we accepted 98% volume retention instead of 97.5%? What would the LR be?" In the spreadsheet, you run another scenario. But the frontier — the full curve of all achievable (LR, volume) combinations — is invisible. You are showing points, not the curve.

**Problem 3: You cannot audit it.** The FCA under Consumer Duty (PS 22/9, effective July 2023) can ask you to show your methodology for every rate decision. "We tried several combinations and chose one that looked sensible" does not satisfy a section 166 request. A formally stated optimisation problem with documented constraints and a reproducible solver does.

### What the optimiser does

The `rate-optimiser` library takes your data, your demand model, and your constraints, and finds the factor adjustment vector that:

1. Achieves the LR target
2. Keeps volume above the floor
3. Satisfies ENBP for every renewal policy
4. Keeps each factor within the approved movement caps
5. Does all of the above with the smallest total disruption to customer premiums

"Smallest total disruption" is formalised as the minimum-dislocation objective, which we explain in Part 4. The solver is SLSQP (Sequential Least Squares Programming) from SciPy. The output is the factor adjustment vector plus shadow prices, an efficient frontier, and an audit trail.