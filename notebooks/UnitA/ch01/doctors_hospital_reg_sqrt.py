# /// notebook
# # ch01 · Example 1.7 – sqrt(MDs) ~ Hospitals
#
# Predict sqrt(MDs) from number of hospitals. A sqrt-transform
# of the response stabilises variance.
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
    from stat2lib.plotting import plot_lm_res2
    from stat2lib.stats import fit_lm
    import marimo as mo
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plot_lm_res2, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=138,
        name="CountyHealth",
        question="Doctors–hospital relationship with sqrt transform",
        feature=["County", "MDs", "Hospitals", "Beds"],
    )
    mo.md(f"## {desc.question} (sqrt transform)")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and sqrt-transform."""
    df = load_rda(desc.name)
    df = df.with_columns(pl.col("MDs").sqrt().alias("sqrt_MDs"))
    df.head(8)
    return df


@app.cell
def _(df, fit_lm):
    """2. Fit model: sqrt(MDs) ~ Hospitals."""
    model = fit_lm(df, "sqrt_MDs ~ Hospitals")
    print(model.summary())
    return model


@app.cell
def _(df, model, plot_lm_res2, plt):
    """3. Diagnostic plots (squared fitted values)."""
    _fig = plot_lm_res2(data=df, xlabel="Hospitals", ylabel="sqrt_MDs", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()