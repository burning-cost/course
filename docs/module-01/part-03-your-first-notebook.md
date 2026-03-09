## Part 3: Your first notebook

### Creating a notebook

In the left sidebar, click **Workspace**. You will see a folder structure. There may be a folder called "Users" with your email address as a subfolder - click into it.

Click the **Add** or **+** button (it may be in the top right, or appear when you hover over a folder). Select **Notebook**.

A dialog box appears. Give the notebook a name - something like "module-01-getting-started" is fine. The default language is Python, which is what we want. Click **Create**.

The notebook opens. You will see:

- A toolbar at the top with buttons for running cells, connecting to a cluster, and so on
- A cluster selector, probably showing "Detached" if the cluster is not yet running
- One empty cell in the main area, ready for code

If the cluster selector shows "Detached", click it and select your cluster from the dropdown. Once connected, the selector shows the cluster name in green.

### What a cell is

A notebook is made of **cells**. Each cell contains code (or markdown text). You run cells one at a time, or all at once.

Click on the empty cell in your notebook. Type:

```python
print("Hello from Databricks")
```

To run the cell, press **Shift+Enter**. The cell runs and the output appears immediately below it:

```text
Hello from Databricks
```

A new empty cell appears below, ready for the next thing you want to try.

This is the core of how notebooks work. You write code in a cell, run it, see the output, write more code in the next cell. The Python session is continuous - variables defined in one cell are available in all subsequent cells, as long as you run them in order.

If you close the notebook and reopen it later, you will need to run the cells again from the top - the Python session does not persist between sessions.

### Basic Python reminder

If you use Python regularly, skip this section. If you mostly use R or Excel, here is the minimum you need for the rest of this module.

**Variables and arithmetic:**

```python
# Variables
claim_count = 42
exposure_years = 1.0
claim_frequency = claim_count / exposure_years

print(f"Claim frequency: {claim_frequency}")
```

Run this (Shift+Enter). The output is `Claim frequency: 42.0`.

The `f"..."` is an f-string - it lets you put variable values inside a string by wrapping them in `{}`.

**Lists and loops:**

```python
accident_years = [2020, 2021, 2022, 2023, 2024]

for year in accident_years:
    print(f"Processing accident year {year}")
```

**Importing libraries:**

```python
import polars as pl
import matplotlib.pyplot as plt
```

The `import` statement loads a library. Libraries need to be installed before you can import them - more on that below.

### Installing a library

Databricks notebooks use a special command for installing libraries: `%pip`. The `%` prefix tells Databricks that this is a notebook magic command, not Python code.

In a new cell, type:

```python
%pip install polars
```

Run it. You will see a stream of output as pip downloads and installs Polars. At the end it says:

```text
Note: you may need to restart the Python kernel to use updated packages.
```

After a `%pip install`, you need to restart the Python kernel before you can use the newly installed library. Run this in the next cell:

```python
dbutils.library.restartPython()
```

This restarts the Python interpreter. Any variables you defined before this point are gone - the session resets. This is normal and expected after installing a library. From now on, put all `%pip install` commands at the very top of your notebook, before any other code, so you only need to restart once.

To install multiple libraries at once:

```python
%pip install polars catboost matplotlib
```

Note: in normal Python development outside Databricks you would use `uv add` or `pip install` from a terminal. Inside Databricks notebooks, `%pip` is the right tool. `uv` is not available inside notebook cells.

### Your first DataFrame with Polars

Create a new cell and run this:

```python
import polars as pl

# A simple DataFrame - like a spreadsheet but in Python
df = pl.DataFrame({
    "accident_year": [2020, 2021, 2022, 2023, 2024],
    "claim_count":   [412,  389,  441,  398,  427],
    "exposure":      [8500, 8750, 9100, 9200, 9400],
})

df
```

The last line - just `df` with no print statement - displays the DataFrame as a formatted table in the notebook output. You will see five rows, three columns, with the data you defined.

Try some basic operations in the next cell:

```python
# Add a calculated column: claim frequency per policy-year
df = df.with_columns(
    (pl.col("claim_count") / pl.col("exposure")).alias("claim_freq")
)

df
```

The `.with_columns()` method adds new columns. `pl.col("claim_count")` refers to the existing column, and `.alias("claim_freq")` gives the new column a name.

### A simple plot

Databricks notebooks display matplotlib plots inline, just like Jupyter.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    [str(y) for y in df["accident_year"].to_list()],
    df["claim_freq"].to_list(),
    color="steelblue",
)
ax.set_xlabel("Accident year")
ax.set_ylabel("Claim frequency")
ax.set_title("Motor claim frequency by accident year")
ax.set_ylim(0, 0.07)

plt.tight_layout()
plt.show()
```

Run this. A bar chart appears in the cell output. The `plt.show()` call is what triggers the display.