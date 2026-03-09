## Part 2: Setting up your Free Edition account

### Creating an account

Go to `https://www.databricks.com/try-databricks` in your browser and select **Free Edition** (not the 14-day trial).

You will see a sign-up form. Fill in your name, email, and a password. Use your work email if you intend to eventually connect this to your organisation's Databricks workspace - it makes things simpler later. If you just want to experiment, a personal email is fine.

After submitting the form, you will receive a verification email. Click the link in it. If it does not arrive within five minutes, check your spam folder.

Once verified, you will be asked to choose a cloud provider: AWS, Azure, or Google Cloud. For Free Edition this choice does not affect what you can do - pick whichever you like. We use AWS in the screenshots throughout this course, but the interface is identical on Azure and Google Cloud.

### What you see after first login

After logging in you land on the Databricks home screen. It looks like this:

- **Left sidebar** - the main navigation. The icons from top to bottom are: Home, Workspace, Data, Compute, Workflows, and Settings. We will use most of these.
- **Central panel** - this changes depending on what you have selected in the sidebar. On first login it shows a "Get started" panel with shortcuts to create a notebook, import data, and so on.
- **Top bar** - your username, a help icon, and a search box.

**Note on the sidebar:** In Databricks Runtime 13 and later (including all current Free Edition workspaces), the **Repos** item no longer appears as a separate sidebar entry. Git-backed notebooks are now accessed through **Workspace** - look for the Git folder icon or use the Workspace browser to navigate to your repos. Part 5 covers this in detail.

The most important things to understand at this stage:

**Workspace** is where your notebooks live. Think of it like a folder structure in File Explorer, but in the cloud. When you create a notebook, it goes into your Workspace. Git-backed repos also live here, accessible via the Workspace browser.

**Compute** is where you manage clusters - the actual computers that run your code. Without a running cluster, notebooks cannot execute. We will cover this properly in Part 6.

**Data** is where you browse and manage datasets. For Free Edition users this shows you DBFS (Databricks File System) - the storage layer for your workspace.

### Starting a cluster before anything else

Before a notebook can run, a cluster must be running. On Free Edition, you get one cluster.

Click **Compute** in the left sidebar. If you have never created a cluster, it shows an empty list and a button that says **Create Compute** (or **Create cluster** - the label has changed between versions). Click it.

On the cluster creation screen you will see various configuration options. On Free Edition, most of these are fixed - you cannot choose the instance type or size. What you can choose is the Databricks Runtime version. Select the latest **LTS ML** version (something like "14.3 LTS ML" or similar - LTS means Long Term Support, ML means it includes machine learning libraries). If you are not sure which to pick, choose the one with "LTS ML" in the name.

Click **Create Compute**. The cluster takes 3-5 minutes to start. The status shows "Pending" and then "Running". While it is starting, move on to the next section.
