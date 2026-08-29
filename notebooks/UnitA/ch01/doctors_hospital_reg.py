# /// notebook
# # ch01 · Example 1.7 – Doctors–Hospitals Regression
#
# Predict number of doctors (MDs) from number of hospitals in a county.
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
    from stat2lib.plotting import plot_pair_scatter, plot_lm_res
    from stat2lib.stats import fit_lm
    import marimo as mo
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plot_lm_res, plot_pair_scatter, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=138,
        name="CountyHealth",
        question="Hospital count predicts doctor count",
        feature=["County", "MDs", "Hospitals", "Beds"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(["Hospitals", "MDs"])
    data.head(8)
    return data, df


@app.cell
def _(data, plot_pair_scatter, plt):
    """2. Scatter plot."""
    _fig, _ax = plot_pair_scatter(data, xlabel="Hospitals", ylabel="MDs")
    plt.gca()
    return


@app.cell
def _(data, fit_lm):
    """3. Fit linear model."""
    model = fit_lm(data, "MDs ~ Hospitals")
    print(model.summary())
    return model


@app.cell
def _(data, model, plot_lm_res, plt):
    """4. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Hospitals", ylabel="MDs", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()