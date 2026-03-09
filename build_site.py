"""
Build the MkDocs course site from the raw training-course markdown files.

This script:
1. Splits each module's tutorial.md into per-Part pages
2. Copies exercises.md as a single page per module
3. Generates mkdocs.yml with full navigation
4. Writes the docs/ directory structure

Run from /home/ralph/burning-cost/course-site/:
    python build_site.py
"""

import re
import os
from pathlib import Path

SOURCE = Path("/home/ralph/burning-cost/workspace/training-course")
DOCS = Path("/home/ralph/burning-cost/course-site/docs")

# Module metadata — title and short description for the index page
MODULES = [
    {
        "num": "01",
        "dir": "module-01-databricks-setup",
        "title": "Databricks for Pricing Teams",
        "description": (
            "Set up a production-ready Databricks environment for a pricing team. "
            "Unity Catalog schema design, Delta tables with time travel, scheduled Workflows, "
            "and the FCA Consumer Duty infrastructure requirements."
        ),
        "duration": "3–4 hours",
        "requires": "Free Edition (Workflows require paid workspace)",
    },
    {
        "num": "02",
        "dir": "module-02-glms-python",
        "title": "GLMs in Python: The Bridge from Emblem",
        "description": (
            "Poisson frequency and gamma severity GLMs in Python, built to match Emblem output. "
            "Factor encoding, exposure offsets, actual-versus-expected diagnostics, "
            "Emblem parity validation, and Radar export."
        ),
        "duration": "5–6 hours",
        "requires": "Free Edition",
    },
    {
        "num": "03",
        "dir": "module-03-gbms-catboost",
        "title": "GBMs for Insurance Pricing: CatBoost",
        "description": (
            "CatBoost frequency-severity model on a synthetic motor portfolio. "
            "Walk-forward cross-validation with IBNR buffer, Optuna hyperparameter tuning, "
            "Gini coefficient diagnostics, and MLflow champion-challenger governance."
        ),
        "duration": "4–5 hours",
        "requires": "Free Edition",
    },
    {
        "num": "04",
        "dir": "module-04-shap-relativities",
        "title": "SHAP Relativities: From GBM to Rating Factor Tables",
        "description": (
            "Extract multiplicative rating factor tables from a CatBoost model using SHAP values. "
            "Smoothed curves for continuous features, confidence intervals, GLM benchmark comparison, "
            "and export to Radar/Akur8/Emblem."
        ),
        "duration": "4–5 hours",
        "requires": "Free Edition",
    },
    {
        "num": "05",
        "dir": "module-05-conformal-intervals",
        "title": "Conformal Prediction Intervals",
        "description": (
            "Distribution-free prediction intervals with guaranteed coverage — no distributional assumptions. "
            "Calibrate and validate a conformal predictor on a CatBoost Tweedie model. "
            "Applications for underwriting referral flags and reserve range estimates."
        ),
        "duration": "4 hours",
        "requires": "Free Edition",
    },
    {
        "num": "06",
        "dir": "module-06-credibility-bayesian",
        "title": "Credibility and Bayesian Pricing: The Thin-Cell Problem",
        "description": (
            "Bühlmann-Straub credibility (EPV/VHM/K structural parameters), "
            "connection to empirical Bayes, and hierarchical Bayesian frequency modelling with PyMC. "
            "Shrinkage plots, posterior credibility factors, and Unity Catalog storage."
        ),
        "duration": "5–6 hours",
        "requires": "Free Edition (Bayesian section requires 16 GB RAM)",
    },
    {
        "num": "07",
        "dir": "module-07-rate-optimisation",
        "title": "Constrained Rate Optimisation",
        "description": (
            "Replace ad-hoc Excel rate exercises with formally stated constrained optimisation. "
            "Loss ratio target, volume floor, per-factor movement caps, ENBP constraint, "
            "efficient frontier analysis and shadow pricing."
        ),
        "duration": "4–5 hours",
        "requires": "Paid Databricks workspace",
    },
    {
        "num": "08",
        "dir": "module-08-end-to-end-pipeline",
        "title": "End-to-End Pricing Pipeline",
        "description": (
            "Capstone module. Full UK personal lines rate review pipeline — 200,000 synthetic motor policies, "
            "shared transform layer, walk-forward CV, CatBoost models, conformal intervals, "
            "constrained rate optimisation, and Consumer Duty compliance audit record."
        ),
        "duration": "6–8 hours",
        "requires": "Paid Databricks workspace",
    },
    {
        "num": "09",
        "dir": "module-09-demand-elasticity",
        "title": "Demand Modelling and Price Elasticity",
        "description": (
            "Conversion and retention modelling with CatBoost. "
            "Causal price elasticity via Double Machine Learning (EconML CausalForestDML), "
            "heterogeneous CATE estimates, profit-maximising price identification, "
            "and ENBP-constrained renewal pricing optimisation."
        ),
        "duration": "5–6 hours",
        "requires": "Paid Databricks workspace",
    },
    {
        "num": "10",
        "dir": "module-10-interactions",
        "title": "Interaction Detection",
        "description": (
            "Automated GLM interaction detection using Combined Actuarial Neural Networks (CANN) "
            "and Neural Interaction Detection (NID). Bonferroni correction, likelihood-ratio tests, "
            "GLM rebuild with discovered interactions, and SHAP interaction validation."
        ),
        "duration": "4–5 hours",
        "requires": "Free Edition",
    },
    {
        "num": "11",
        "dir": "module-11",
        "title": "Model Monitoring and Drift Detection",
        "description": (
            "Detect when a deployed pricing model degrades. "
            "Population stability index, characteristic stability index, actual-versus-expected ratios "
            "with confidence intervals, Gini drift z-test, automated traffic-light triggers, "
            "Delta Lake logging, and Databricks job scheduling for continuous monitoring."
        ),
        "duration": "4–5 hours",
        "requires": "Free Edition (scheduling exercises require paid workspace)",
    },
    {
        "num": "12",
        "dir": "module-12-spatial-territory",
        "title": "Spatial Territory Rating",
        "description": (
            "Replace Emblem-style postcode group rating with Bayesian spatial models. "
            "Adjacency matrix construction, Moran's I spatial autocorrelation test, "
            "BYM2 model fitted via PyMC, territory relativity extraction, "
            "and integration into a downstream GLM rating engine."
        ),
        "duration": "5–6 hours",
        "requires": "Paid Databricks workspace",
    },
]


def slugify(text: str) -> str:
    """Convert a Part title to a filename-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text[:60].rstrip("-")


def find_real_part_headers(content: str) -> list:
    """
    Find all ## Part N: headers that are NOT inside code blocks.
    Returns list of (char_offset, title_string) tuples.
    """
    lines = content.splitlines(keepends=True)
    in_code_block = False
    char_pos = 0
    results = []
    part_pattern = re.compile(r"^## (Part \d+[a-z]?:.*)$")
    fence_pattern = re.compile(r"^```")

    for line in lines:
        if fence_pattern.match(line.rstrip()):
            in_code_block = not in_code_block
        elif not in_code_block:
            m = part_pattern.match(line.rstrip())
            if m:
                results.append((char_pos, m.group(1).strip()))
        char_pos += len(line)

    return results


def split_tutorial(content: str, module_num: str) -> list[dict]:
    """
    Split a tutorial.md into a list of {title, slug, content} dicts.
    Each dict is one page — either the intro before the first Part,
    or a Part section.

    Only considers ## Part N: headers that appear outside code blocks.
    When the same part number appears more than once (tutorial prose +
    notebook %md cell pattern), uses only the first occurrence.
    """
    pages = []
    matches = find_real_part_headers(content)

    if not matches:
        pages.append({"title": "Tutorial", "slug": "tutorial", "content": content})
        return pages

    # Deduplicate by part number — keep only the first occurrence of each number
    seen_part_nums: set = set()
    deduped = []
    part_num_re = re.compile(r"^Part (\d+[a-z]?):")
    for offset, title in matches:
        m = part_num_re.match(title)
        if m:
            pnum = m.group(1)
            if pnum not in seen_part_nums:
                seen_part_nums.add(pnum)
                deduped.append((offset, title))
        else:
            deduped.append((offset, title))

    matches = deduped

    # Intro content before first Part header
    intro_content = content[: matches[0][0]].strip()
    if intro_content:
        pages.append({"title": "Overview", "slug": "overview", "content": intro_content})

    for i, (start, part_title) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(content)
        part_content = content[start:end].strip()
        part_content = part_content.rstrip().rstrip("-").rstrip()

        m = part_num_re.match(part_title)
        if m:
            part_num = m.group(1).zfill(2)
            rest = re.sub(r"^Part \d+[a-z]?:\s*", "", part_title)
            slug = f"part-{part_num}-{slugify(rest)}"
        else:
            slug = slugify(part_title)

        pages.append({"title": part_title, "slug": slug, "content": part_content})

    return pages


def convert_admonitions(content: str) -> str:
    """
    Convert common informal callout patterns to MkDocs Material admonitions.

    Patterns handled:
    - > **Note:** ... -> !!! note
    - > **Warning:** ... -> !!! warning
    - > **Tip:** ... -> !!! tip
    - **Note:** at start of paragraph -> !!! note inline
    """
    # Blockquote-style: > **Note:** text
    def convert_blockquote(m):
        kind_raw = m.group(1).lower()
        kind_map = {
            "note": "note",
            "warning": "warning",
            "warn": "warning",
            "tip": "tip",
            "important": "important",
            "caution": "caution",
            "info": "info",
        }
        kind = kind_map.get(kind_raw, "note")
        body_lines = m.group(0).splitlines()
        # Strip "> " prefix and "**Kind:**" from first line
        inner = []
        first = True
        for line in body_lines:
            stripped = line.lstrip("> ").rstrip()
            if first:
                stripped = re.sub(r"^\*\*\w+:\*\*\s*", "", stripped)
                first = False
            if stripped:
                inner.append("    " + stripped)
        return f"!!! {kind}\n" + "\n".join(inner)

    content = re.sub(
        r"^> \*\*(Note|Warning|Warn|Tip|Important|Caution|Info):\*\*[^\n]*(?:\n^>[^\n]*)*",
        convert_blockquote,
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    return content


def write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_module_docs(module: dict) -> list[tuple[str, str]]:
    """
    Build all pages for one module. Returns list of (nav_title, relative_path)
    pairs for mkdocs.yml navigation.
    """
    mod_num = module["num"]
    mod_dir = module["dir"]
    mod_title = module["title"]
    source_dir = SOURCE / mod_dir
    docs_dir = DOCS / f"module-{mod_num}"

    nav_entries = []

    # --- Tutorial pages ---
    tutorial_path = source_dir / "tutorial.md"
    if tutorial_path.exists():
        content = tutorial_path.read_text(encoding="utf-8")
        content = convert_admonitions(content)
        pages = split_tutorial(content, mod_num)

        for page in pages:
            out_path = docs_dir / f"{page['slug']}.md"
            write_page(out_path, page["content"])
            rel_path = f"module-{mod_num}/{page['slug']}.md"
            nav_entries.append((page["title"], rel_path))

    # --- Exercises page ---
    exercises_path = source_dir / "exercises.md"
    if exercises_path.exists():
        ex_content = exercises_path.read_text(encoding="utf-8")
        ex_content = convert_admonitions(ex_content)
        out_path = docs_dir / "exercises.md"
        write_page(out_path, ex_content)
        nav_entries.append(("Exercises", f"module-{mod_num}/exercises.md"))

    return nav_entries


def build_index() -> str:
    """Build the course landing page."""
    lines = [
        "# Modern Insurance Pricing with Python and Databricks",
        "",
        "A hands-on course for UK pricing actuaries and analysts. Twelve modules taking you from Databricks basics to production-grade pricing pipelines — GLMs, GBMs, SHAP relativities, conformal prediction, Bayesian credibility, constrained rate optimisation, demand elasticity, and spatial territory models.",
        "",
        "---",
        "",
        "## What this course covers",
        "",
        "The course is structured around the problems pricing teams actually face. Each module tackles one problem in depth, with a full tutorial, working code, and exercises with solutions.",
        "",
        "All code runs on Databricks. Most modules run on the free tier. Where a paid workspace is needed, it is noted clearly.",
        "",
        "---",
        "",
        "## Modules",
        "",
    ]

    for mod in MODULES:
        num = mod["num"]
        title = mod["title"]
        desc = mod["description"]
        duration = mod["duration"]
        requires = mod["requires"]

        lines += [
            f"### Module {int(num)}: {title}",
            "",
            desc,
            "",
            f"**Duration:** {duration} &nbsp;&nbsp; **Requires:** {requires}",
            "",
            f"[Go to Module {int(num)} &rarr;](module-{num}/overview.md){{.md-button}}",
            "",
            "---",
            "",
        ]

    lines += [
        "## How to use this course",
        "",
        "Work through the modules in order — each one builds on the previous. If you are already comfortable with Databricks, you can start at Module 2.",
        "",
        "Each module follows the same pattern:",
        "",
        "1. Read the tutorial, working through the code in your Databricks notebook",
        "2. Complete the exercises — they are not optional, they are where the learning happens",
        "3. Check your solutions against the provided answers",
        "",
        "The code in every module runs against synthetic data that is generated within the notebook itself. You do not need access to your insurer's data to complete the course.",
        "",
        "---",
        "",
        "## Setup",
        "",
        "Before starting Module 1, read the [Getting Started](getting-started.md) guide. It covers Databricks account setup, library installation, and how the notebooks are organised.",
        "",
    ]

    return "\n".join(lines)


def build_getting_started() -> str:
    gs_path = SOURCE / "GETTING-STARTED.md"
    if gs_path.exists():
        return gs_path.read_text(encoding="utf-8")
    return "# Getting Started\n\nSee the course README for setup instructions.\n"


def generate_mkdocs_yml(all_nav: dict) -> str:
    """Generate the mkdocs.yml content."""
    nav_lines = ["nav:"]
    nav_lines.append("  - Home: index.md")
    nav_lines.append("  - Getting Started: getting-started.md")

    for mod in MODULES:
        num = mod["num"]
        title = mod["title"]
        entries = all_nav.get(num, [])
        if not entries:
            continue

        nav_lines.append(f"  - 'Module {int(num)}: {title}':")
        for entry_title, entry_path in entries:
            # Escape single quotes in titles for YAML
            safe_title = entry_title.replace("'", "''")
            nav_lines.append(f"    - '{safe_title}': {entry_path}")

    nav_str = "\n".join(nav_lines)

    return f"""site_name: Burning Cost
site_description: Modern Insurance Pricing with Python and Databricks
site_author: Burning Cost
site_url: https://burning-cost.github.io/course/

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  font:
    text: Inter
    code: JetBrains Mono
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.expand
    - navigation.top
    - navigation.footer
    - search.highlight
    - search.share
    - content.code.copy
    - content.code.annotate
    - toc.integrate

plugins:
  - search:
      lang: en

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.tabbed:
      alternate_style: true
  - tables
  - toc:
      permalink: true
  - attr_list
  - md_in_html

extra:
  generator: false

{nav_str}
"""


def main():
    print("Building course site...")
    DOCS.mkdir(parents=True, exist_ok=True)

    # Index page
    index_content = build_index()
    (DOCS / "index.md").write_text(index_content, encoding="utf-8")
    print("  Written: docs/index.md")

    # Getting started
    gs_content = build_getting_started()
    (DOCS / "getting-started.md").write_text(gs_content, encoding="utf-8")
    print("  Written: docs/getting-started.md")

    # All modules
    all_nav = {}
    for mod in MODULES:
        num = mod["num"]
        print(f"  Processing Module {num}: {mod['title']}")
        nav_entries = build_module_docs(mod)
        all_nav[num] = nav_entries
        print(f"    {len(nav_entries)} pages")

    # mkdocs.yml
    yml_content = generate_mkdocs_yml(all_nav)
    (Path("/home/ralph/burning-cost/course-site") / "mkdocs.yml").write_text(
        yml_content, encoding="utf-8"
    )
    print("  Written: mkdocs.yml")

    print("\nDone. Run 'mkdocs build' or 'mkdocs serve' from /home/ralph/burning-cost/course-site/")


if __name__ == "__main__":
    main()
