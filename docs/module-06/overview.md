# Module 6: Credibility and Bayesian Methods — The Thin-Cell Problem

In Module 5 you built conformal prediction intervals — honest uncertainty bands around every prediction your model makes. You know how to quantify prediction uncertainty for individual risks. This module is about a different problem, one that exists before the model is even fitted: what do you do when you do not have enough data to trust the observed rate?

This is the thin-cell problem. It is not a niche concern. Every UK personal lines pricer working with postcode districts, vehicle groups, or affinity schemes encounters it every week. By the end of this module you will know the two principled tools for handling it — Bühlmann-Straub credibility and Bayesian hierarchical models — and you will have implemented both from scratch in a Databricks notebook.

[Download the notebook for this module](notebook.py)

---