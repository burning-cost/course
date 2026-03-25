## Part 12: Setting up alerts

### Why email notifications are not enough

The Databricks job notifications from Part 11 tell you the job ran and whether it succeeded or failed. They do not tell you whether the monitoring results showed anything concerning. A job can succeed (no errors) and produce a red traffic light (model has drifted significantly). You need a separate mechanism to alert on the *content* of the results, not just the execution status.

There are two ways to implement content-based alerting in Databricks: Databricks SQL Alerts and notebook-based notifications using the email API. We cover both.

### Approach 1: Databricks SQL Alerts

Databricks SQL Alerts run a SQL query on a schedule and send a notification if the query returns results. They are the right tool when you want to alert based on data in a Delta table - which is exactly what we have after Part 10.

In your Databricks workspace, go to **SQL** in the left sidebar, then **Alerts**, then **Create Alert**.

Configure the alert query:

```sql
SELECT
    run_date,
    model_name,
    ae_ratio,
    recommendation,
    psi_score
FROM main.motor_monitoring.monitoring_log
WHERE
    recommendation IN ('RECALIBRATE', 'REFIT', 'INVESTIGATE', 'MONITOR_CLOSELY')
    AND run_date >= DATE_SUB(CURRENT_DATE(), 7)
ORDER BY run_date DESC
```

Alert condition: **Has rows** (trigger the alert if the query returns any rows).

Schedule: run once per day (the alert will only trigger in the 7 days following a monitoring run, because that is the window in the WHERE clause).

Notification: add the pricing team email address.

This gives you a daily check on whether any recent monitoring run produced an amber or red result. If everything is green, the query returns no rows and no alert fires.

### A more targeted alert: red flags only

Create a second alert that escalates immediately to the head of pricing for red results:

```sql
SELECT
    run_date,
    model_name,
    ae_ratio,
    ae_ci_lower,
    ae_ci_upper,
    gini_cur,
    psi_score,
    recommendation
FROM main.motor_monitoring.monitoring_log
WHERE
    recommendation = 'REFIT'
    AND run_date >= DATE_SUB(CURRENT_DATE(), 3)
```

Set this alert to run every hour (or every 30 minutes if you want faster escalation). A red flag should not sit unnoticed for a day.

### Approach 2: Programmatic alerts from the notebook

If you want to send a richer notification with context - not just "the alert fired" but the actual metric values and a recommendation - you can send email directly from the monitoring notebook using a webhook or SMTP call.

For Databricks environments with a configured SMTP gateway:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_monitoring_alert(report_dict: dict, model_name: str, current_date: str, recipients: list[str]) -> None:
    """Send an email alert when the monitoring report requires action.

    Parameters
    ----------
    report_dict : dict
        Output of MonitoringReport.to_dict(). Keys: results, recommendation, murphy_available.
    model_name : str
        Model name string for the email subject.
    current_date : str
        Monitoring period end date (e.g. "2024-06-30").
    recipients : list[str]
        Email addresses to notify.
    """
    recommendation = report_dict["recommendation"]

    if recommendation == "NO_ACTION":
        return  # No alert needed

    subject = f"[{recommendation}] Motor model monitoring - {current_date}"

    results = report_dict["results"]
    ae   = results["ae_ratio"]
    gini = results["gini"]

    body = f"""
Model monitoring report for {model_name}
Period ending: {current_date}
Recommendation: {recommendation}

METRICS:
  A/E ratio:  {ae['value']:.4f}  (95% CI: [{ae['lower_ci']:.4f}, {ae['upper_ci']:.4f}])  [{ae['band']}]
  Gini (ref): {gini['reference']:.4f}
  Gini (cur): {gini['current']:.4f}  p={gini['p_value']:.4f}  [{gini['band']}]

TOP CSI FLAGS:
"""
    if "csi" in results:
        for item in sorted(results["csi"], key=lambda x: x["csi"], reverse=True)[:5]:
            body += f"  {item['feature']:<25} CSI={item['csi']:.4f}  [{item['band']}]\n"

    body += f"""
This is an automated alert from the Burning Cost monitoring pipeline.
Notebook: module-11-model-monitoring
"""

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = "monitoring@yourcompany.com"
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(body, "plain"))

    # Replace with your SMTP server details
    # Store credentials in Databricks Secrets, not here
    smtp_host = dbutils.secrets.get(scope="monitoring", key="smtp_host")
    smtp_port = 587
    smtp_user = dbutils.secrets.get(scope="monitoring", key="smtp_user")
    smtp_pass = dbutils.secrets.get(scope="monitoring", key="smtp_pass")

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(msg["From"], recipients, msg.as_string())

    print(f"Alert sent to {recipients}")


# Call this after generating the report summary
ALERT_RECIPIENTS = [
    "pricing@yourcompany.com",
    "head.of.pricing@yourcompany.com",
]

send_monitoring_alert(report.to_dict(), MODEL_NAME, CURRENT_DATE, ALERT_RECIPIENTS)
```

Store `smtp_host`, `smtp_user`, and `smtp_pass` in Databricks Secrets (under the scope `monitoring`). Never put credentials in a notebook.

### Approach 3: Microsoft Teams or Slack webhook

If your team uses Teams or Slack, a webhook notification is often more effective than email because it reaches people where they already work:

```python
import requests
import json

def send_teams_alert(report_dict: dict, model_name: str, current_date: str, webhook_url: str) -> None:
    """Post a monitoring summary card to a Teams channel.

    Parameters
    ----------
    report_dict : dict
        Output of MonitoringReport.to_dict(). Keys: results, recommendation, murphy_available.
    model_name : str
        Model name for the card title.
    current_date : str
        Monitoring period end date.
    webhook_url : str
        Teams incoming webhook URL.
    """
    recommendation = report_dict["recommendation"]

    if recommendation == "NO_ACTION":
        return

    colour = "FF0000" if recommendation == "REFIT" else "FFA500"

    results = report_dict["results"]
    ae = results["ae_ratio"]

    card = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": colour,
        "summary": f"Model monitoring {recommendation}: {model_name}",
        "sections": [{
            "activityTitle": f"Model monitoring: {recommendation}",
            "activitySubtitle": (
                f"{model_name} - period ending {current_date}"
            ),
            "facts": [
                {"name": "A/E ratio",  "value": f"{ae['value']:.4f} [{ae['band']}]"},
                {"name": "Gini (cur)", "value": f"{results['gini']['current']:.4f}  [{results['gini']['band']}]"},
            ],
        }],
    }

    response = requests.post(
        webhook_url,
        data=json.dumps(card),
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"Teams alert failed: {response.status_code} {response.text}")
    else:
        print("Teams alert sent.")


teams_webhook = dbutils.secrets.get(scope="monitoring", key="teams_webhook_url")
send_teams_alert(report.to_dict(), MODEL_NAME, CURRENT_DATE, teams_webhook)
```

### Choosing your alert strategy

Use SQL Alerts as the primary mechanism. They are low-maintenance, require no code to update, and are visible to anyone with access to the Databricks SQL workspace. The programmatic email/Teams approach supplements this: use it in the monitoring notebook for richer context when the overall result is amber or red.

Do not rely solely on programmatic alerts from the notebook. If the notebook fails before reaching the alert code, no alert fires. SQL Alerts query the Delta table, so they fire correctly even if a notebook run has to be re-run from a checkpoint.
