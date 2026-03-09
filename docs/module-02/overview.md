# Module 2: GLMs in Python - The Bridge from Emblem

In Module 1 you created a Databricks account, started a cluster, ran your first Python cells, and saved data to a Delta table. This module builds on that foundation. By the end of it you will have fitted a working Poisson frequency GLM and a Gamma severity GLM on a realistic motor dataset, extracted factor relativities, run diagnostics, and seen how the Python output compares to what Emblem would produce.

This is not a statistics lecture. We assume you already understand what a GLM is, what IRLS does, and what a relativity means. The goal here is to show you that the same model you have been running in Emblem can be reproduced in Python - with better auditability, better version control, and a cleaner integration with the rest of a modern pricing stack.

---

## Why this is worth doing

Let us be clear about something before we start. Emblem fits GLMs correctly. It uses IRLS, the same algorithm Python's statsmodels uses. It handles exposure offsets. It produces deviance statistics, factor charts, and actual-versus-expected plots. If you are a pricing actuary who has been fitting frequency-severity models in Emblem for ten years, you are not doing it wrong.

The problem is the infrastructure around the model. The Emblem project file is not version-controlled. Nobody commits it to Git. The data extract you fed it lives on a network drive with a name like `motor_extract_final_v2_ACTUAL.csv`. When the FCA asks you to reproduce the relativities from your Q3 2023 renewal cycle, you go hunting for the right combination of software version, project file, and data extract - and you hope nothing has changed.

This matters now more than it did five years ago. PS 21/5 (the general insurance pricing practices rules, effective January 2022) banned price walking and introduced explicit audit trail requirements for pricing decisions. Consumer Duty (PS 22/9, effective July 2023) extended this further to require demonstrable fair value - meaning the FCA wants to walk into your office, ask about any price charged to any customer in the past three years, and have you show them the model, the inputs, and the decision trail in under an hour.

Moving your GLM to Python and Databricks solves this. The model code is version-controlled. The training data is a Delta table with time travel. The fitted model is logged to MLflow with the parameters, metrics, and artefacts. Running the model from six months ago means checking out the relevant Git tag and pointing at the Delta table at that timestamp. That is reproducibility you can demonstrate to a regulator.

The GLM itself is not the hard part. The encoding, the validation, and the export are where the work is. That is what this module covers.

---