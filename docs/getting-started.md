# Getting Started — Modern Insurance Pricing with Python and Databricks

Welcome. You have purchased the complete Burning Cost training course. This guide covers what is included, what you need before you start, and how to get up and running.

---

## What is included

The course is eight modules. Each module contains:

- **README.md** — a concise overview of the module: what it covers, prerequisites, estimated time, and what you will be able to do when you finish
- **tutorial.md** — the main written content. Read this first, before opening the notebook. It explains the reasoning behind every design decision, not just the mechanics
- **notebook.py** — a Databricks notebook covering the same material end-to-end on synthetic data. Import and run this as you work through the tutorial
- **exercises.md** — hands-on exercises per module with full worked solutions

| Module | Title | Estimated time |
|--------|-------|----------------|
| 01 | Databricks for Pricing Teams | 3–4 hours |
| 02 | GLMs in Python — The Bridge from Emblem | 4–5 hours |
| 03 | GBMs for Insurance Pricing — CatBoost | 5–6 hours |
| 04 | SHAP Relativities — From GBM to Rating Factor Tables | 3–4 hours |
| 05 | Conformal Prediction Intervals | 4–5 hours |
| 06 | Credibility & Bayesian Pricing — The Thin-Cell Problem | 4–5 hours |
| 07 | Constrained Rate Optimisation | 3–4 hours |
| 08 | End-to-End Pricing Pipeline | 5–6 hours |

Total: approximately 31–39 hours of study and practical work, depending on how deeply you engage with the exercises.

---

## Your purchase includes all future content

Your £295 purchase gives you lifetime access to this course. We update modules as tools evolve and add new content over time. All future additions are included at no extra charge. This includes any new modules we publish and access to all future Burning Cost tools and content delivered through this platform.

Re-download at any time to get the latest version. If you spot an error — in a notebook or a tutorial — report it to pricing.frontier@gmail.com and we will fix it and notify all course purchasers.

---

## Where to start

Start with Module 01. It sets up the Databricks environment — the Unity Catalog schema, Delta table conventions, and MLflow experiment structure — that every subsequent module depends on. If you skip it, the later modules will work, but you will have to adapt the setup steps yourself.

Work through the modules in order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

Module 08 (End-to-End Pipeline) is the capstone. It assumes you are comfortable with everything that came before.

---

## Prerequisites

**Databricks.** All notebooks run on Databricks. Databricks Free Edition (community.cloud.databricks.com) is sufficient for Modules 01–05 and the classical credibility section of Module 06. The Bayesian section of Module 06 requires at least 16 GB RAM — a paid workspace with a single-node cluster is needed for that section. Module 01's Workflows exercises and Modules 07–08 also require a paid workspace.

**Python.** You need to be able to read Python code and follow a data pipeline. You do not need to be a software engineer. If you have used Python for data work — loading a CSV, filtering a DataFrame, calling a function — you have enough Python for this course.

**GLMs.** Modules 02 onwards assume you are comfortable with frequency-severity GLMs: you know what a link function is, what `exp(beta)` gives you, and what a factor table looks like. You do not need to know Python GLM libraries before you start.

**Polars.** The course uses Polars for all data manipulation. If you have only used Pandas, the syntax is similar enough that the code will make sense as you go — there are inline comments wherever Polars idioms differ from what a Pandas user would expect. You do not need prior Polars experience.

---

## How to import notebooks into Databricks

Each module's `notebook.py` is a Databricks source-format notebook — a Python file with `# COMMAND ----------` cell delimiters. Import it as follows:

1. Log into your Databricks workspace.
2. In the left sidebar, click **Workspace**.
3. Navigate to the folder where you want to store the notebook.
4. Click the **kebab menu** (three dots) next to the folder name and select **Import**.
5. In the import dialog, select **File** and drag the `.py` file onto the target area, or click to browse and select it.
6. Databricks will import the file as a notebook. It will appear in the folder with a notebook icon.
7. Open the notebook and attach it to a running cluster before executing any cells.

Run cells with Shift+Enter or use **Run all** from the toolbar. The notebooks are designed to run top-to-bottom without modification on first pass — they generate their own synthetic data.

---

## Open-source libraries

Several modules use open-source pricing libraries built by Burning Cost. Install them with pip before running the relevant notebooks.

### Option A — Install all libraries at once (recommended)

On Databricks, run this in a notebook cell before working through the course:

```python
%pip install insurance-cv shap-relativities insurance-conformal credibility bayesian-pricing rate-optimiser --quiet
```

Then restart the Python kernel:

```python
dbutils.library.restartPython()
```

### Option B — Install per module

Each notebook includes an install cell at the top that installs only what that module needs. Run that cell first, then restart the kernel before executing any other cells.

**Libraries by module:**

| Module | Libraries needed |
|--------|-----------------|
| 03 | `catboost`, `optuna`, `mlflow`, `insurance-cv` |
| 04 | `shap-relativities`, `catboost` |
| 05 | `insurance-conformal`, `catboost` |
| 06 | `credibility`, `bayesian-pricing`, `pymc`, `arviz` |
| 07 | `rate-optimiser` |
| 08 | All of the above |

The `bayesian-pricing` package requires PyMC 5.x and ArviZ. The install takes a few minutes. Run the install cell first, then restart the Python kernel (`dbutils.library.restartPython()`) before executing any other cells.

### Source code

Full source code for each library is available from Burning Cost. These are open-source tools — read, modify, and extend them as needed for your own pricing workflows.

---

## Support

Email **pricing.frontier@gmail.com** with questions. Include which module and which section you are working on. Response within 2 business days.
