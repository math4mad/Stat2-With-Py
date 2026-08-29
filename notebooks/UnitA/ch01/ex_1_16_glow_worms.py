# /// notebook
# # ch01 · Ex 1.16 – Glow Worms
#
# Female glow worms attract males by glowing; brighter lanterns
# may lead to more eggs. Linear regression of Eggs on Lantern size.
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
        name="GlowWorms",
        question="Lantern size vs egg count",
        feature=["Lantern", "Eggs"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda):
    """1. Load data."""
    data = load_rda(desc.name)
    data.head(8)
    return data


@app.cell
def _(data, fit_lm):
    """2. Fit model: Eggs ~ Lantern."""
    model = fit_lm(data, "Eggs ~ Lantern")
    print(model.summary())
    return model


@app.cell
def _(data, model, plot_lm_res, plt):
    """3. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Lantern", ylabel="Eggs", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()