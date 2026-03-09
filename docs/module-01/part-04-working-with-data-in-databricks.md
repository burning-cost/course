## Part 4: Working with data in Databricks

### Uploading a CSV file

In real pricing work, your data will come from a database or a file upload. Here is how to upload a CSV to your Databricks workspace.

First, create a small CSV on your local machine. Open a text editor (or use Excel and save as CSV) and create a file called `sample_policies.csv` with this content:

```python
policy_ref,inception_year,exposure_years,claim_count,area_band,ncd_years,vehicle_group,driver_age
POL00001,2023,1.0,0,B,5,15,42
POL00002,2023,0.75,1,A,0,22,27
POL00003,2023,1.0,0,C,3,18,55
POL00004,2023,0.5,0,D,2,31,34
POL00005,2023,1.0,1,B,1,19,29
POL00006,2023,1.0,0,A,5,12,61
POL00007,2023,0.25,0,E,4,25,38
POL00008,2023,1.0,0,C,3,17,45
POL00009,2023,1.0,2,B,0,28,22
POL00010,2023,1.0,0,A,5,11,58
```

Save the file.

Now, in Databricks, click **Data** in the left sidebar. You will see options for browsing tables and files.

Look for a button or option that says **Add data** or **Upload file** - the exact label varies slightly by Databricks version. Click it.

You will be prompted to upload a file. Drag and drop your `sample_policies.csv`, or use the file browser to find it.

After uploading, Databricks will show you a preview of the file and suggest a table name. For now, note the path where the file was uploaded - it will be something like `/FileStore/tables/sample_policies.csv` or similar. We will read it from that path.

Alternatively, you can create the data directly in the notebook without uploading a file - which is more reliable for this exercise:

```python
import polars as pl

# Create sample policy data directly in the notebook
policies = pl.DataFrame({
    "policy_ref":     ["POL00001", "POL00002", "POL00003", "POL00004", "POL00005",
                       "POL00006", "POL00007", "POL00008", "POL00009", "POL00010"],
    "inception_year": [2023] * 10,
    "exposure_years": [1.0, 0.75, 1.0, 0.5, 1.0, 1.0, 0.25, 1.0, 1.0, 1.0],
    "claim_count":    [0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
    "area_band":      ["B", "A", "C", "D", "B", "A", "E", "C", "B", "A"],
    "ncd_years":      [5, 0, 3, 2, 1, 5, 4, 3, 0, 5],
    "vehicle_group":  [15, 22, 18, 31, 19, 12, 25, 17, 28, 11],
    "driver_age":     [42, 27, 55, 34, 29, 61, 38, 45, 22, 58],
})

policies
```

### Basic data exploration

These are the first things you should do with any new dataset:

```python
# How many rows and columns?
print(f"Shape: {policies.shape}")  # (rows, columns)

# What are the column names and types?
print(policies.dtypes)
print(policies.columns)
```

```python
# First few rows
policies.head(5)
```

```python
# Summary statistics for numeric columns
policies.describe()
```

The `.describe()` output shows count, mean, standard deviation, min, and max for each numeric column. For a real claims dataset, the minimum claim count should never be negative and the exposure should always be positive - these are the first sanity checks to run.

```python
# Count claims by area band
policies.group_by("area_band").agg(
    pl.col("claim_count").sum().alias("total_claims"),
    pl.col("exposure_years").sum().alias("total_exposure"),
    pl.len().alias("policy_count"),
).with_columns(
    (pl.col("total_claims") / pl.col("total_exposure")).alias("claim_freq")
).sort("area_band")
```

This is a basic one-way analysis: claim frequency by area band. It is the same calculation you would do in a spreadsheet pivot table, but it scales to millions of rows without slowing down.

### Saving as a Delta table

Delta Lake is the data storage format that makes Databricks different from just a Python notebook environment. A Delta table is like a database table, but stored in your cloud workspace, with the ability to query historical versions.

To save a Polars DataFrame as a Delta table, we go via Spark (the distributed computing engine underneath Databricks):

```python
# Convert Polars to a Spark DataFrame, then write as Delta
spark.createDataFrame(policies.to_pandas()).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("policies_sample")
```

There are three things happening here:

1. `policies.to_pandas()` - converts the Polars DataFrame to a pandas DataFrame (an intermediate step)
2. `spark.createDataFrame(...)` - converts the pandas DataFrame to a Spark DataFrame
3. `.write.format("delta").mode("overwrite").saveAsTable("policies_sample")` - writes it to a Delta table

The `spark` variable is already available in any Databricks notebook — you do not need to import or create it. Spark is the distributed computing engine that Databricks runs on; it handles reading and writing data to Delta tables. You interact with it through this `spark` variable.

Why the `.to_pandas()` step? Spark does not natively understand Polars DataFrames, so we convert to pandas as an intermediate format that Spark can ingest. This is a one-line overhead you will see throughout the course.

You should see a confirmation message. The table is now saved and will persist even if you close the notebook or the cluster shuts down.

**What is Delta Lake?** It is a storage format built on top of Parquet (a compressed, columnar file format). What makes it special is that every write creates a new version of the table, and you can query any version at any point in time. If someone accidentally deletes data, or if you need to prove what the data looked like when you ran a model three months ago, Delta time travel gives you that. We will use this extensively in Module 2.

### Reading the table back

```python
# Read the Delta table back into a Polars DataFrame
policies_from_delta = spark.table("policies_sample").toPandas()
df_back = pl.from_pandas(policies_from_delta)

df_back.head()
```

Or, using SQL directly in a notebook cell with the `%sql` magic:

```sql
%sql
SELECT area_band, COUNT(*) as policy_count, SUM(claim_count) as total_claims
FROM policies_sample
GROUP BY area_band
ORDER BY area_band
```

The `%sql` magic lets you run SQL directly in a notebook cell. The result displays as a table. This is useful for quick data queries without writing Python.