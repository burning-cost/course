# Module 11: Exposure Curves and Increased Limits Factors

In Module 6 you learned how to blend thin experience with portfolio priors using credibility and Bayesian hierarchical models. In Module 7 you built a formally constrained rate optimiser that satisfies loss ratio targets, volume floors, and FCA fair-pricing rules simultaneously. Both of those modules deal with the ground-up insurance product: you are pricing the first layer of loss, the part the direct insurer pays.

This module moves up the tower. We are concerned with what happens above a retention: per-risk excess-of-loss reinsurance, London market layers, and the increased limits factors used to extend a basic-limit rate to higher policy limits. These problems share a common foundation that the direct pricing curriculum typically never teaches: the exposure curve.

Every Lloyd's syndicate, every reinsurance underwriter, and every commercial lines pricing team that prices above a deductible needs this. Most UK pricing actuaries who came up through personal lines have never seen the maths. Those who have seen it usually learned it in R, from the `mbbefd` package, or in Excel using Swiss Re tables from 2005. There has been no Python implementation on PyPI until recently.

We use the `insurance-ilf` library throughout. It provides the MBBEFD distribution class, curve fitting, ILF tables, and per-risk XL pricing in a single tested package. By the end of this module you will understand what exposure curves are, where the MBBEFD family comes from, and how to fit, validate, and deploy them for real pricing work on Databricks.

By the end of this module you will have:

- Understood what an exposure curve is and why it is the right tool for excess-of-loss pricing
- Derived the key MBBEFD formulas from first principles, accessibly
- Installed and explored the Swiss Re standard curves (Y1-Y4, Lloyd's)
- Fitted MBBEFD distributions to claims data using MLE, with correct handling of truncated and censored data
- Built ILF tables from fitted curves and understood the marginal ILF structure
- Priced a per-risk XL layer from a ceding company's risk profile using the exposure rating method
- Produced and interpreted Lee diagrams for visual communication with underwriters
- Understood where this connects to London market practice

---