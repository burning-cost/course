## Part 5: Organising your work

### Notebooks, Repos, and Files

Databricks gives you three places to put code. Understanding the difference saves headaches later.

**Workspace notebooks** are the default. When you created a notebook in Part 3, it went into your Workspace. These notebooks are stored in Databricks itself - they are not files on your computer. They are fine for exploration and learning. The limitation is that they are not version-controlled: if you delete a cell by accident, there is no undo history beyond the session. If you overwrite work, it is gone.

**Repos** are Git-backed. A Repo in Databricks is a connection to a Git repository (on GitHub, Azure DevOps, or Bitbucket). When you edit a notebook in a Repo, you can commit and push your changes just like you would from VS Code or a terminal. This is the right place for any code that matters - model training notebooks, data preparation scripts, production pipelines. We cover setting up a Repo below.

**Files** are arbitrary files in the workspace filesystem - config files, YAML, JSON, CSV. Not notebooks. Less common, but useful for storing configuration alongside your repo code.

The rule of thumb: **exploratory work in Workspace, anything production-ready in a Repo**.

### Connecting a Git repository

You will need a GitHub account (free). If you do not have one, go to `https://github.com` and sign up.

Create a new repository on GitHub:
1. Go to `https://github.com/new`
2. Give it a name, e.g. `databricks-pricing-practice`
3. Set it to Private
4. Check "Add a README file"
5. Click **Create repository**

Copy the repository URL - it looks like `https://github.com/yourusername/databricks-pricing-practice.git`.

Back in Databricks:
1. Click **Repos** in the left sidebar
2. Click **Add Repo** (or **+** > **Repo**)
3. Paste the GitHub URL
4. Databricks will ask for your GitHub credentials - use a Personal Access Token, not your password. To create one: GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token. Give it `repo` scope. Copy the token and paste it into Databricks.
5. Click **Create Repo**

Databricks clones the repository. You will see its contents appear in the Repos panel. You can now create notebooks inside this repo, edit them, and commit the changes back to GitHub.

For the rest of this course, we recommend creating new notebooks inside your Repo rather than in the Workspace. It gives you a safety net and is the professional habit to build.

### Folder structure for a pricing project

Here is a sensible structure for a pricing project repo. You do not need to create all of this now - this is the shape to grow into.

```
motor-pricing/
    notebooks/
        01_data_preparation.py
        02_frequency_model.py
        03_severity_model.py
        04_factor_extraction.py
    src/
        motor/
            features.py
            validation.py
    config/
        model_params.yaml
        base_levels.yaml
    README.md
```

- `notebooks/` - numbered in execution order. Each notebook does one thing.
- `src/` - reusable Python code (functions used across multiple notebooks).
- `config/` - configuration files. Putting model parameters in YAML files rather than hardcoding them in notebooks means you can change parameters without touching the code.

For exploratory work - trying things, one-off analyses, checking something quickly - use your Workspace rather than the repo. The repo should contain code that you would be comfortable showing to a colleague or a regulator.