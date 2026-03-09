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
    overall_traffic_light,
    psi_score
FROM main.motor_monitoring.monitoring_log
WHERE
    overall_traffic_light IN ('AMBER', 'RED')
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
    overall_traffic_light
FROM main.motor_monitoring.monitoring_log
WHERE
    overall_traffic_light = 'RED'
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

def send_monitoring_alert(summary: dict, recipients: list[str]) -> None:
    """Send an email alert when the monitoring report is AMBER or RED."""

    traffic_light = summary["overall_traffic_light"]

    if traffic_light == "GREEN":
        return  # No alert needed

    subject = f"[{traffic_light}] Motor model monitoring - {summary['current_date']}"

    ae = summary["metrics"]["ae_ratio"]
    gini = summary["metrics"]["gini"]
    psi = summary["metrics"]["psi_score"]

    body = f"""
Model monitoring report for {summary['model_name']}
Period: {summary['reference_date']} to {summary['current_date']}
Status: {traffic_light}

METRICS:
  A/E ratio:  {ae['value']:.4f}  (95% CI: [{ae['ci_lower']:.4f}, {ae['ci_upper']:.4f}])  {ae['traffic_light']}
  Score PSI:  {psi['value']:.4f}  {psi['traffic_light']}
  Gini (ref): {gini['gini_ref']:.4f}
  Gini (cur): {gini['gini_cur']:.4f}  p={gini['p_value']:.4f}  {gini['traffic_light']}

TOP CSI FLAGS:
"""
    for item in sorted(summary["csi"], key=lambda x: x["csi"], reverse=True)[:5]:
        body += f"  {item['feature']:<25} CSI={item['csi']:.4f}  {item['traffic_light']}\n"

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

send_monitoring_alert(summary, ALERT_RECIPIENTS)
```

Store `smtp_host`, `smtp_user`, and `smtp_pass` in Databricks Secrets (under the scope `monitoring`). Never put credentials in a notebook.

### Approach 3: Microsoft Teams or Slack webhook

If your team uses Teams or Slack, a webhook notification is often more effective than email because it reaches people where they already work:

```python
import requests
import json

def send_teams_alert(summary: dict, webhook_url: str) -> None:
    """Post a monitoring summary card to a Teams channel."""

    traffic_light = summary["overall_traffic_light"]

    if traffic_light == "GREEN":
        return

    colour = "FF0000" if traffic_light == "RED" else "FFA500"

    ae = summary["metrics"]["ae_ratio"]

    card = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": colour,
        "summary": f"Model monitoring {traffic_light}: {summary['model_name']}",
        "sections": [{
            "activityTitle": f"Model monitoring: {traffic_light}",
            "activitySubtitle": (
                f"{summary['model_name']} - period ending {summary['current_date']}"
            ),
            "facts": [
                {"name": "A/E ratio",  "value": f"{ae['value']:.4f} ({ae['traffic_light']})"},
                {"name": "Score PSI",  "value": f"{summary['metrics']['psi_score']['value']:.4f}"},
                {"name": "Gini (cur)", "value": f"{summary['metrics']['gini']['gini_cur']:.4f}"},
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
send_teams_alert(summary, teams_webhook)
```

### Choosing your alert strategy

Use SQL Alerts as the primary mechanism. They are low-maintenance, require no code to update, and are visible to anyone with access to the Databricks SQL workspace. The programmatic email/Teams approach supplements this: use it in the monitoring notebook for richer context when the overall result is amber or red.

Do not rely solely on programmatic alerts from the notebook. If the notebook fails before reaching the alert code, no alert fires. SQL Alerts query the Delta table, so they fire correctly even if a notebook run has to be re-run from a checkpoint.
