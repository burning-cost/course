# Module 1 Exercises

Six exercises, one for each substantive part of the tutorial. All six can be completed in Free Edition - no paid workspace required.

Work through them in order. Each builds slightly on the previous. Solutions are at the end of each exercise.

---

## Exercise 1: Make sense of what Databricks is doing

**Covers:** Part 1 - What is Databricks and why should you care

**Objective.** Ground your understanding of what problem Databricks solves before touching any code.

**Tasks:**

Think about your current pricing workflow - or if you are new to pricing, think about the description in Part 1. Answer these four questions in writing (a few sentences each). There is no code involved.

1. **The version problem.** Your team runs a motor model. Six months later, a claims manager asks: "Did the model use the updated NCD relativity that was changed in the April data refresh?" How would you answer that question today? What would change if the training data were stored as a Delta table with version history?

2. **The reproducibility problem.** A junior analyst produces a frequency model. They leave the team. The model needs to be re-run with new data. What are the three most likely failure modes that prevent the re-run from working, and how does a shared Databricks environment with a Git-backed repo mitigate each one?

3. **The audit problem.** Consumer Duty requires you to demonstrate that your pricing model was trained on specific data, validated by a named actuary, and approved before deployment. Map each of those requirements to a specific Databricks feature from Part 1.

4. **What it does not replace.** Name two things that Databricks does not replace in a typical UK personal lines pricing workflow, and explain why not.

---

### Discussion — Exercise 1

There are no single right answers, but here are the key points:

**Question 1.** With scripts on a laptop and data on SharePoint, the honest answer is probably "I would have to check the email chain from April and hope someone noted which file they used". With Delta time travel, you look up the table version number logged at model training time (which a well-structured audit record would store) and query the table at that version: `SELECT * FROM pricing.motor.claims_exposure VERSION AS OF 3`. The data that produced that model is exactly recoverable.

**Question 2.** The three most common failure modes when re-running someone else's model:
- The script assumes libraries at specific versions that have since changed (mitigated by a shared cluster with fixed library versions)
- The script reads from a file path that no longer exists (mitigated by Delta tables with stable names)
- The script contains undocumented parameters or magic numbers that the original analyst adjusted by hand (mitigated by version-controlled code with config files, and by MLflow experiment logging)

**Question 3.**
- "Trained on specific data" - Delta time travel + logging the table version at training time
- "Validated by a named actuary" - model version tags in MLflow Model Registry with an `approved_by` field
- "Approved before deployment" - setting the `production` alias in MLflow Model Registry, gated by a human step

**Question 4.** Databricks does not replace Radar (the factor table deployment system) or Emblem (the traditional GLM modelling tool, where teams still use it for regulatory familiarity). Radar is where factor tables live and are applied to quotes - Databricks produces the factor tables but does not deploy them to the rating engine. Emblem is a separate tool with a different workflow; some teams retain it for certain models where regulatory expectation is for a traditional GLM, run in Emblem, with Emblem's output documentation.

---

## Exercise 2: Get a cluster running and run your first code

**Covers:** Parts 2 and 3 - Setting up your account and your first notebook

**Objective.** Go from a blank Databricks environment to a working notebook with real output.

**Tasks:**

1. Create a Free Edition account at `databricks.com/try-databricks` if you have not already. Start a cluster (Compute > Create Compute > select the LTS ML runtime). Wait for it to reach Running status.

2. Create a notebook in your Workspace called `exercise-02`. Attach it to your running cluster.

3. Run these three cells in order. Confirm each produces the expected output.

**Cell 1:**
```python
print("Cluster is running")
print(f"Python version: {__import__('sys').version}")
```
Expected: two lines of output. The Python version should be 3.10 or 3.11.

**Cell 2:**
```python
%pip install polars matplotlib
```
Expected: a stream of installation output, ending with a note about restarting the kernel.

**Cell 3:** (run this after Cell 2 completes)
```python
dbutils.library.restartPython()
```
Expected: the kernel restarts. Any variables from previous cells are now gone.

4. After the restart, in a new cell, reproduce the claim frequency bar chart from Part 3. Use this data:

```python
import polars as pl
import matplotlib.pyplot as plt

df = pl.DataFrame({
    "accident_year": [2019, 2020, 2021, 2022, 2023],
    "claim_count":   [3820, 3241, 3890, 4102, 3956],
    "exposure":      [72000, 68000, 74000, 79000, 81000],
})
```

The chart should show claim frequency (claim_count / exposure) for each accident year. Label the axes and give it a title.

5. Notice that claim frequency dropped in 2020 and has risen since. In one sentence, what is the plausible real-world explanation for the 2020 dip?

---

### Solution — Exercise 2

```python
import polars as pl
import matplotlib.pyplot as plt

df = pl.DataFrame({
    "accident_year": [2019, 2020, 2021, 2022, 2023],
    "claim_count":   [3820, 3241, 3890, 4102, 3956],
    "exposure":      [72000, 68000, 74000, 79000, 81000],
})

df = df.with_columns(
    (pl.col("claim_count") / pl.col("exposure")).alias("claim_freq")
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    [str(y) for y in df["accident_year"].to_list()],
    df["claim_freq"].to_list(),
    color="steelblue",
)
ax.set_xlabel("Accident year")
ax.set_ylabel("Claim frequency")
ax.set_title("Motor claim frequency by accident year (exercise data)")
ax.set_ylim(0, 0.07)
plt.tight_layout()
plt.show()
```

**Question 5.** The 2020 dip is almost certainly Covid-19 lockdowns - significantly reduced vehicle usage meant fewer accidents, so claim frequency fell sharply. This is a well-documented pattern across UK motor books. The subsequent rise reflects normalising driving behaviour plus the inflation effect on claim counts from increased repair costs prompting more claims.

---

## Exercise 3: Explore a dataset

**Covers:** Part 4 - Working with data in Databricks

**Objective.** Load a small simulated claims dataset, run basic exploration, and save it as a Delta table.

**Setup.** Run this in a new notebook (or continue in your exercise-02 notebook after the `%pip install` and restart):

```python
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
n = 500

claims = pl.DataFrame({
    "policy_ref":     [f"POL{i:05d}" for i in range(n)],
    "accident_year":  rng.choice([2021, 2022, 2023], size=n).tolist(),
    "exposure_years": rng.uniform(0.1, 1.0, size=n).round(3).tolist(),
    "claim_count":    rng.poisson(0.09, size=n).tolist(),
    "incurred":       (rng.poisson(0.09, size=n) * rng.exponential(2800, size=n)).round(2).tolist(),
    "area_band":      rng.choice(["A", "B", "C", "D", "E"], size=n).tolist(),
    "ncd_years":      rng.integers(0, 6, size=n).tolist(),
    "vehicle_group":  rng.integers(1, 51, size=n).tolist(),
    "driver_age":     rng.integers(17, 85, size=n).tolist(),
    "annual_mileage": rng.integers(3000, 25000, size=n).astype(float).tolist(),
})
```

**Tasks:**

1. Use `.shape`, `.columns`, `.dtypes`, and `.describe()` to understand the dataset. How many rows and columns? What is the mean claim frequency (total claims / total exposure)?

2. Identify any data quality issues. Specifically, check:
   - Are there any rows where `incurred > 0` but `claim_count == 0`? (An incurred cost with no claim - suspicious)
   - Are there any rows where `claim_count > 0` but `incurred == 0`? (A claim with no cost - also suspicious)

   Report the counts.

3. Produce a one-way frequency table grouped by `area_band`. Show: policy count, total claims, total exposure, and claim frequency. Sort by area band.

4. Save the dataset as a Delta table called `exercise_claims`. Read it back and confirm the row count matches.

5. Using `%sql`, run a SQL query directly in a notebook cell to show the average `annual_mileage` for drivers aged under 30 vs 30 and over.

---

### Solution — Exercise 3

```python
# Task 1: Basic exploration
print(f"Shape: {claims.shape}")
print(f"\nColumns: {claims.columns}")
print(f"\nData types:\n{claims.dtypes}")
print(f"\nDescribe:\n{claims.describe()}")

total_claims = claims["claim_count"].sum()
total_exposure = claims["exposure_years"].sum()
print(f"\nMean claim frequency: {total_claims / total_exposure:.4f}")
```

```python
# Task 2: Data quality
n_incurred_no_claim = int(((claims["incurred"] > 0) & (claims["claim_count"] == 0)).sum())
n_claim_no_incurred = int(((claims["claim_count"] > 0) & (claims["incurred"] == 0)).sum())

print(f"Incurred > 0 but claim_count == 0: {n_incurred_no_claim}")
print(f"Claim_count > 0 but incurred == 0: {n_claim_no_incurred}")
```

The incurred values in the setup code are generated independently of claim_count (to simulate real data quality issues), so you will likely find some. In practice, `incurred > 0, claim_count == 0` can mean a claim was paid and then the count field was not updated - a data extract issue worth flagging to the claims team.

```python
# Task 3: One-way by area band
oneway = (
    claims
    .group_by("area_band")
    .agg(
        pl.len().alias("policy_count"),
        pl.col("claim_count").sum().alias("total_claims"),
        pl.col("exposure_years").sum().round(1).alias("total_exposure"),
    )
    .with_columns(
        (pl.col("total_claims") / pl.col("total_exposure")).round(4).alias("claim_freq")
    )
    .sort("area_band")
)

oneway
```

```python
# Task 4: Save as Delta table
spark.createDataFrame(claims.to_pandas()).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("exercise_claims")

# Read back and verify
count_back = spark.table("exercise_claims").count()
print(f"Original rows: {len(claims)}")
print(f"Rows in Delta table: {count_back}")
assert count_back == len(claims), "Row count mismatch - check write"
print("Row counts match.")
```

```sql
%sql
SELECT
    CASE WHEN driver_age < 30 THEN 'Under 30' ELSE '30 and over' END AS age_group,
    ROUND(AVG(annual_mileage), 0) AS avg_annual_mileage,
    COUNT(*) AS policy_count
FROM exercise_claims
GROUP BY age_group
ORDER BY age_group
```

---

## Exercise 4: Set up a Git repo and organise your work

**Covers:** Part 5 - Organising your work

**Objective.** Connect a GitHub repository to Databricks and commit your first notebook.

**Tasks:**

1. Create a free GitHub account if you do not have one. Create a new private repository called `databricks-pricing-practice`. Initialise it with a README.

2. Create a Personal Access Token on GitHub (Settings > Developer settings > Personal access tokens > Generate new token). Give it `repo` scope. Copy the token - you will not see it again after leaving the page.

3. In Databricks, go to **Repos** in the left sidebar. Click **Add Repo**. Paste your repository URL. Enter your GitHub username and the Personal Access Token as the password. Create the repo.

4. Inside the repo in Databricks, create a folder called `notebooks`. Inside that, create a new notebook called `01_data_exploration`.

5. Copy the one-way frequency table code from Exercise 3 into this notebook. Add a markdown cell at the top (change the cell type from Code to Markdown in the dropdown) with a heading and a sentence describing what the notebook does.

   To add a markdown cell: click **+ Code** to create a new cell, then change the dropdown from "Python" to "Markdown". A markdown heading looks like `# My Heading`.

6. Commit the notebook to GitHub. In the Repos panel, look for a **Git** button or the branch indicator at the top of the Repos view. Click it to open the Git dialogue. Stage the new file and write a commit message (e.g. "Add initial data exploration notebook"). Push to main.

7. Go to your GitHub repository in a browser and confirm the notebook file appears there.

---

### Discussion — Exercise 4

The key habit being built here is: anything that matters goes into Git. A notebook committed to GitHub can be recovered, reviewed by a colleague, and reproduced on a different Databricks workspace. A notebook that lives only in your Workspace cannot.

In a real team setting, you would also set up branch protection on GitHub (so changes to the main branch require a pull request review) and potentially connect the repo to a Databricks Repo in a shared workspace, so the whole team works from the same code base.

The notebook format Databricks uses for Python files in Repos is `.py` with special comment markers - it looks like a Python script but Databricks renders it as a notebook. When you view it on GitHub it shows as readable Python code, not as a JSON blob (which is what Jupyter `.ipynb` files look like on GitHub). This makes code review much easier.

---

## Exercise 5: Manage your cluster

**Covers:** Part 6 - Compute basics

**Objective.** Understand the cluster lifecycle and what persists across cluster restarts.

**Tasks:**

1. In a notebook, run this code to create a variable and a Delta table:

```python
import polars as pl

# In-memory variable
my_variable = "This lives in memory"

# Delta table
data = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
spark.createDataFrame(data.to_pandas()).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("exercise_cluster_test")

print(f"Variable: {my_variable}")
print("Delta table written.")
```

2. Go to **Compute** and terminate your cluster. Wait for it to reach Terminated status.

3. Restart the cluster (click Start). Wait for it to reach Running status. Reattach your notebook to the cluster.

4. In a new cell, try to access `my_variable`:

```python
print(my_variable)
```

What happens? Run it and note the error.

5. In the next cell, try to read the Delta table:

```python
import polars as pl

df = pl.from_pandas(spark.table("exercise_cluster_test").toPandas())
print(df)
```

What happens? Does this work?

6. Write one sentence explaining the difference in behaviour between questions 4 and 5. This is the practical lesson about what to always persist to Delta.

---

### Solution — Exercise 5

**Question 4.** You get a `NameError: name 'my_variable' is not defined`. The Python session was completely reset when the cluster restarted. All in-memory variables, all imported libraries, all intermediate DataFrames - gone.

**Question 5.** The Delta table read works fine. You get the DataFrame back with three rows, exactly as you saved it. The table is stored on disk (in your workspace's cloud storage), not in memory. A cluster restart does not affect it.

**Question 6.** In-memory Python variables exist only for the life of the cluster session. Delta tables are persistent storage on disk and survive cluster restarts, workspace closures, and anything else. Any result that you might need after the current session must be saved to a Delta table - never rely on an in-memory DataFrame surviving beyond the notebook run.

This is particularly important for long model training runs. If you train a CatBoost model on a large dataset (which might take 20 minutes), save the model artefact (via MLflow or to a file) and save the predictions to a Delta table before doing anything else. If the cluster dies mid-analysis, you do not want to retrain the model.

---

## Exercise 6: Putting it together - a mini pipeline

**Covers:** All parts

**Objective.** Build a minimal end-to-end pipeline: generate data, clean it, save it, analyse it, and produce a chart - all in a single structured notebook.

This exercise is more open-ended than the previous ones. There is no single correct answer.

**Tasks:**

Create a new notebook in your Repos folder called `02_mini_pipeline`. Structure it with markdown headings for each section:

1. **Setup** - `%pip install` for any needed libraries, then `dbutils.library.restartPython()`, then imports.

2. **Generate data** - Create a simulated motor policy dataset with these fields: `policy_ref`, `accident_year` (2021-2023), `exposure_years`, `claim_count`, `incurred`, `area_band` (A-F), `ncd_years` (0-5), `driver_age`. Use 2,000 rows. Use `np.random.default_rng(99)` as the random seed so results are reproducible.

3. **Clean data** - Apply these rules and print the count of rows removed by each:
   - Remove rows with `exposure_years <= 0`
   - Set `incurred = 0` where `claim_count == 0` (a claim with zero cost is plausible; incurred with no claim is not)
   - Remove rows where `driver_age < 17` or `driver_age > 100`

4. **Save to Delta** - Write the cleaned dataset to a table called `exercise_pipeline_claims`. Print the row count before and after.

5. **Analyse from Delta** - Read the table back. Compute claim frequency and average severity by `ncd_years`. Display as a Polars DataFrame.

6. **Chart** - Plot claim frequency by NCD years as a bar chart. You should expect to see decreasing frequency with increasing NCD, since more experienced drivers have fewer claims. If the simulated data does not show this pattern, that is fine - note it and explain why random data would not necessarily show real-world patterns.

7. **Commit** - Commit this notebook to your GitHub repository with the message "Add mini pipeline notebook - exercise 6".

---

### Reference solution outline — Exercise 6

A complete solution is not provided here, because building it yourself is the point. But here are the key code patterns you will need:

**For data generation with the random seed:**
```python
rng = np.random.default_rng(99)
n = 2000
```

**For Poisson-distributed claim counts (realistic for motor):**
```python
claim_count = rng.poisson(0.09, size=n).tolist()
```

**For conditionally generated incurred (only where there is a claim):**
```python
import numpy as np
counts = np.array(rng.poisson(0.09, size=n))
incurred = np.where(counts > 0, rng.exponential(2800, size=n), 0.0).round(2).tolist()
```

**For the cleaning step - counting removed rows:**
```python
original_count = len(df)
df_clean = df.filter(pl.col("exposure_years") > 0)
removed = original_count - len(df_clean)
print(f"Removed {removed} rows with non-positive exposure")
```

**For the one-way analysis from the Delta table:**
```python
result = (
    df_from_delta
    .group_by("ncd_years")
    .agg(
        pl.col("claim_count").sum().alias("total_claims"),
        pl.col("exposure_years").sum().alias("total_exposure"),
        pl.col("incurred").filter(pl.col("claim_count") > 0).mean().alias("avg_severity"),
    )
    .with_columns(
        (pl.col("total_claims") / pl.col("total_exposure")).round(4).alias("claim_freq")
    )
    .sort("ncd_years")
)
```

If the bar chart shows a clear decreasing trend with NCD years - that is a signal in randomly generated data that would need explaining. Random data with Poisson claim counts and no NCD effect baked in should show roughly flat frequencies across NCD years. Real data shows a pronounced trend because higher NCD reflects driver experience and risk selection. When you use real data in Module 2, the pattern will be clear.
