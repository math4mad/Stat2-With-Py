# /// notebook
# # ch02 · Book Pages vs Price
#
# Linear regression of textbook price on number of pages.
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
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import plot_lm_res
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plot_lm_res, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=253,
        name="TextPrices",
        question="pages-prices-reg",
        feature=["Pages", "Price"],
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
    """2. Fit model: Price ~ Pages."""
    model = fit_lm(data, "Price ~ Pages")
    print(model.summary())
    return model


@app.cell
def _(data, model, plot_lm_res, plt):
    """3. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Pages", ylabel="Price", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()