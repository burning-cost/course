## Part 11: Scheduling monitoring as a Databricks job

> **Databricks tier note:** Parts 11 and 12 cover features that require a paid Databricks workspace. Workflows/Jobs, SQL Alerts, and Secrets (`dbutils.secrets`) are not available on Databricks Community Edition. If you are on Community Edition, read these parts for understanding of how production monitoring works — but you will not be able to run them. The notebook itself (Parts 1–10) runs on Community Edition without modification.

### Why manual runs are not enough

Running the monitoring notebook manually each month is fine for learning. It is not acceptable for a deployed pricing model. Manual runs get skipped when people are on holiday. They get run late. They get run on the wrong date range. Most importantly, they do not produce alerts when something goes wrong at 2am on a Sunday.

Databricks Jobs turn a notebook into a scheduled process. The job runs at a defined time, on a defined cluster, with defined parameters, and sends notifications if it fails or if the results breach a threshold. You get the monitoring run done consistently whether or not anyone remembers to do it.

### Preparing the notebook for job execution

Before scheduling, the notebook needs to be job-ready. This means:

1. No interactive inputs - all parameters come from the configuration cell at the top
2. Errors must propagate - do not catch exceptions silently; let the job fail visibly
3. The run date must be computed from `date.today()` - not hard-coded

Update the configuration cell to use dynamic dates:

```python
from datetime import date
from dateutil.relativedelta import relativedelta

# In job execution, CURRENT_DATE is the last day of the previous month.
# In interactive execution, you may override this manually.
today = date.today()
first_of_this_month = today.replace(day=1)
CURRENT_DATE = str(first_of_this_month - relativedelta(days=1))  # last day of previous month

# Reference date: 12 months before current date (rolling 12-month reference window)
REFERENCE_DATE = str(first_of_this_month - relativedelta(months=13) - relativedelta(days=1))

print(f"Monitoring window: {REFERENCE_DATE} -> {CURRENT_DATE}")
print(f"Run date:          {str(today)}")
```

This means every monthly run automatically uses the previous full month as the current period and the 12 months before that as the reference period. You never need to update the dates manually.

### Creating the job in the Databricks UI

In your Databricks workspace:

1. Click **Workflows** in the left sidebar
2. Click **Create Job**
3. Name the job `motor-model-monitoring-monthly`

In the task configuration:

- **Task type**: Notebook
- **Source**: Workspace (or Git if you have a repo connected)
- **Path**: Select your `module-11-model-monitoring` notebook
- **Cluster**: Select your existing cluster, or configure a new job cluster

For a monitoring job, a job cluster (created fresh for each run) is preferable to an all-purpose cluster. Job clusters are cheaper (they only exist for the duration of the job) and more reliable (no residual state from previous notebook runs). Configure the job cluster with:

- **Single node**: fine for monitoring (no distributed computation needed for the metrics themselves)
- **Runtime**: latest LTS version
- **Libraries**: add `insurance-monitoring`, `insurance-datasets`, `catboost`, `polars` under **Libraries** in the cluster config

### Configuring the schedule

Under **Schedules**, click **Add schedule**:

- **Schedule type**: Cron
- **Cron expression**: `0 6 1 * *` - this runs at 06:00 on the 1st of every month
- **Timezone**: Europe/London

Why the 1st of the month at 06:00 UK time? The monitoring window ends on the last day of the previous month. Running on the 1st ensures the data for the previous month is available. Running at 06:00 means the results are ready when the pricing team arrives in the morning, and any failures surface early in the working day rather than at the end.

### Adding email notifications

Under **Notifications**, add your email (and the head of pricing's email) for:

- **On start**: optional
- **On success**: recommended - confirms the run completed
- **On failure**: mandatory - alerts you immediately if the run fails

Add an email address in each box. Databricks will send a notification for each event.

### Running the job manually to test

Before relying on the schedule, run the job manually to verify it works end to end:

1. Click **Run now** in the job page
2. Watch the run in the **Runs** tab
3. Click into the run to see the notebook output
4. Verify the Delta tables have been updated

If the run fails, the error will appear in the run output. Common issues:

- **Library not found**: add the library to the job cluster configuration, not just the all-purpose cluster
- **Table not found**: check the catalog and schema names in the configuration cell match what exists in Unity Catalog
- **Model not found**: verify the model name and version in the configuration cell match the MLflow Model Registry entry

### Parameterising the job for ad-hoc runs

Sometimes you need to run monitoring for a specific historical period. Databricks Jobs support notebook parameters via `dbutils.widgets`. Add this to the configuration cell:

```python
# Job parameters - these can be passed when triggering the job via API or UI.
# If not passed (interactive run), fall back to the computed defaults.
dbutils.widgets.text("current_date_override", "")
dbutils.widgets.text("reference_date_override", "")

current_date_override = dbutils.widgets.get("current_date_override")
reference_date_override = dbutils.widgets.get("reference_date_override")

if current_date_override:
    CURRENT_DATE = current_date_override
    print(f"Using override: CURRENT_DATE = {CURRENT_DATE}")

if reference_date_override:
    REFERENCE_DATE = reference_date_override
    print(f"Using override: REFERENCE_DATE = {REFERENCE_DATE}")
```

When you trigger the job via the Databricks REST API (for example, to backfill a missing month), you can pass `{"current_date_override": "2024-03-31"}` as the run parameters.

### Triggering via the REST API

For integration with other systems (e.g., triggering the monitoring job after the monthly data refresh completes), use the Jobs REST API:

```python
import requests

WORKSPACE_URL = "https://your-workspace.azuredatabricks.net"  # change this
TOKEN = dbutils.secrets.get(scope="monitoring", key="databricks_pat")
JOB_ID = 12345  # get this from the job URL in the UI

response = requests.post(
    f"{WORKSPACE_URL}/api/2.1/jobs/run-now",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "job_id": JOB_ID,
        "notebook_params": {
            "current_date_override": "2024-03-31",
        },
    },
)

run_id = response.json()["run_id"]
print(f"Job triggered. Run ID: {run_id}")
print(f"Monitor at: {WORKSPACE_URL}/#job/{JOB_ID}/run/{run_id}")
```

Store your PAT in Databricks Secrets rather than hard-coding it. The `dbutils.secrets.get()` call retrieves it at runtime without exposing it in the notebook.
