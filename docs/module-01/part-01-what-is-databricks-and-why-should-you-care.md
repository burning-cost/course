## Part 1: What is Databricks and why should you care

### The problem it solves

Most pricing teams work like this: the model lives in a collection of R or Python scripts on someone's laptop. When it needs to run, that person runs it. The training data is a CSV somewhere on SharePoint - possibly the right version, possibly not. The last model run was six months ago and nobody is quite sure what version of the data it used. When someone asks "what would happen if we changed the NCD relativity?" the answer involves finding the right person, hoping their environment still works, and waiting.

This is not a criticism. It is a natural consequence of tools designed for individual use being stretched to support team workflows.

Databricks is a cloud platform designed to fix exactly this. It provides:

- **A shared compute environment** - everyone runs code on the same infrastructure, with the same library versions, without needing to set up anything on their own machine
- **Notebooks that live in the cloud** - open them from any browser, share them with a link, no local installation required
- **Versioned data storage** - the data you trained the model on is preserved, with a version number, and you can query it exactly as it was on any date
- **Scheduled pipelines** - the model can run itself on the 15th of each month, email you if something goes wrong, and not require anyone to be at their desk

The reason pricing teams specifically are moving to Databricks is regulatory pressure. Consumer Duty requires insurers to demonstrate that pricing is consistent with the intended risk segmentation and does not produce unfair outcomes. That demonstration requires being able to show, for any given model: what data it was trained on, when it ran, what it produced, and who approved it. A collection of scripts on individual laptops cannot show any of that. Databricks can.

### What it replaces in your workflow

| What you do now | What Databricks replaces it with |
|---|---|
| Scripts on a laptop | Notebooks in a shared workspace |
| CSV files on SharePoint | Delta tables with version history |
| "Email me the output" | Shared notebooks and scheduled exports |
| Manual model runs | Workflows (automated pipelines) |
| "Which data did we use?" | Delta time travel - query any past version |
| Library conflicts between team members | Shared cluster with fixed library versions |

It does not replace Radar or Emblem. The output of a Databricks pipeline - factor tables, GLM relativities, pure premiums - still feeds into Radar for deployment. Databricks is where you build and run the analysis. Radar is where the output lives.

### Free Edition vs paid

Databricks offers a Free Edition at `databricks.com/try-databricks`. It is real Databricks - the same notebooks, the same Python environment, the same Delta Lake storage format. The limitations are:

- Single small cluster (no choice of size)
- No Unity Catalog (the governance layer - covered in Part 7)
- No scheduled Workflows
- No multi-user workspace (it is a single-person environment)

Free Edition is sufficient for everything in Parts 2 through 6 of this module, and for most of Module 2. When we reach features that require a paid workspace, we will say so explicitly.

For a team production environment, your insurer will have a paid workspace on Azure or AWS - set up by the platform or data engineering team. This module uses Free Edition so you can follow along without needing to ask for access to a shared environment.