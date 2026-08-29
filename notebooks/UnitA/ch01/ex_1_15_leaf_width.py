# /// notebook
# # ch01 · Ex 1.15 – Leaf Width Linear Regression
#
# Plant leaf width decreases as environmental temperature rises.
# Explore the linear relationship between year and leaf width.
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
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import plot_lm_res
    from stat2lib.stats import fit_lm
    import marimo as mo
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plot_lm_res, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=178,
        name="LeafWidth",
        question="Year–LeafWidth linear regression",
        feature=["Width", "Length", "LWRatio", "Area", "Year"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(["Year", "Width"])
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Fit model: Width ~ Year."""
    model = fit_lm(data, "Width ~ Year")
    print(model.summary())
    return model


@app.cell
def _(data, model, plot_lm_res, plt):
    """3. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Year", ylabel="Width", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()