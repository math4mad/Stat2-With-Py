# /// notebook
# # ch04 · First Year GPA – Predictor Exploration
#
# Pairplot of GPA predictors: HSGPA, SATV, SATM, HU, SS, GPA.
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    return Path, Stat2Table, load_rda, mo, pl, plt, sns, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=442,
        name="FirstYearGPA",
        question="predictors select",
        feature=["HSGPA", "SATV", "SATM", "HU", "SS", "GPA"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(desc.feature)
    data.describe()
    return data, df


@app.cell
def _(data, plt, sns):
    """2. Pairplot."""
    _pdf = data.to_pandas()
    _fig = sns.pairplot(_pdf, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 30})
    _fig.fig.suptitle("First Year GPA Predictors", y=1.02, fontsize=14)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()