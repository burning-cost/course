## Part 7: Unity Catalog (preview)

You will not use Unity Catalog in Free Edition - it is a paid feature. But understanding what it is now means Part 2 of the course makes sense when you get there.

### What Unity Catalog is

Every Delta table you created in Parts 4 and 6 has a simple name: `policies_sample`. In a team environment with multiple projects, product lines, and data domains, simple names cause problems. Which `claims` table? Is it raw or processed? Which team owns it?

Unity Catalog is Databricks' governance layer. It organises all tables in a three-level hierarchy:

```
catalog . schema . table
```

For example: `pricing.motor.claims_exposure`

- **Catalog** - the top level. Usually corresponds to an environment (production, development) or a business unit.
- **Schema** - the middle level. Usually corresponds to a product line or analytical domain (motor, home, governance).
- **Table** - the specific dataset.

When you reference a table as `pricing.motor.claims_exposure`, you know exactly what it is, who owns it, and where it sits in the organisation's data hierarchy.

### Why it matters for pricing teams

Beyond organisation, Unity Catalog provides:

- **Lineage tracking** - it automatically records which tables fed into which other tables and which notebooks created them. After a model run you can look up a table in the Catalog UI and see the full chain: which source tables it came from, which notebook processed it, which other tables depend on it. This is the audit trail that Consumer Duty requires.

- **Access control** - you can grant read access to `pricing.motor.claims_exposure` to the data science team and write access only to the pricing actuaries, without any manual file permission management.

- **Column-level security** - sensitive columns (policyholder names, dates of birth) can be masked so analysts who do not need PII see redacted values, while those with appropriate access see the real data.

On Free Edition, you work with simple table names (no catalog or schema prefix). When you move to a paid workspace, your platform team will have set up a catalog and schema for your team - they will tell you the prefix to use.

### The catalog hierarchy in practice

In a paid workspace, the first thing to understand is what catalog and schema your team uses. Common patterns:

```
pricing.motor.claims_exposure          <- Motor claims data
pricing.motor.model_relativities       <- Model outputs
pricing.home.claims_exposure           <- Home claims data
pricing.governance.model_run_log       <- Audit log
```

You do not need to create these - in most organisations the platform team sets up the catalog structure and grants your team access. What you do need is the prefix, which goes at the start of every table reference in your notebooks.

This is covered in detail in Module 2, once you are working in a team environment.

---

## What's next

You now have a working Databricks environment. You know how to create notebooks, run Python code, load data, and save it as a Delta table. You understand what a cluster is and how to manage it.

**Module 2: GLMs for Frequency and Severity** - trains Poisson frequency and Gamma severity models on real claims data loaded into Delta tables. Introduces PySpark for reading large datasets, Polars for analysis, and MLflow for experiment tracking. By the end of Module 2 you have a working GLM pipeline that produces factor tables.

**Module 3: Gradient Boosted Models with CatBoost** - replaces the GLM frequency model with a CatBoost model. Covers hyperparameter tuning, cross-validation designed for insurance data, and model comparison against the GLM benchmark.

The infrastructure you set up here - the workspace, the cluster, the Delta tables - is used in every subsequent module. When Module 2 reads from `pricing.motor.claims_exposure`, it is reading from a table structured exactly as described in Part 4 and Part 7.