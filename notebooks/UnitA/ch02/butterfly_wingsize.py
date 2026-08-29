# /// notebook
# # ch02 · Butterfly Wing Size vs Temperature
#
# Linear regression of butterfly wing size on temperature.
# Data filtered for one sex.
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
    from stat2lib.plotting import plot_pair_scatter, plot_lm_res
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plot_lm_res, plot_pair_scatter, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=138,
        name="ButterfliesBc",
        question="wings-temperature-reg",
        feature=["Temp", "Wing", "Sex", "Species"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and filter by sex."""
    df = load_rda(desc.name)
    # Use first sex group (matching Julia: groupby(df, :Sex)[1])
    first_sex = df["Sex"].unique().sort().item(0)
    data = df.filter(pl.col("Sex") == first_sex).select(["Temp", "Wing"])
    print(f"Filtered to Sex = {first_sex}, n = {data.shape[0]}")
    data.head(8)
    return data, df, first_sex


@app.cell
def _(data, plot_pair_scatter, plt):
    """2. Scatter plot."""
    _fig, _ax = plot_pair_scatter(data, xlabel="Temp", ylabel="Wing")
    plt.gca()
    return


@app.cell
def _(data, fit_lm):
    """3. Fit model: Wing ~ Temp."""
    model = fit_lm(data, "Wing ~ Temp")
    print(model.summary())
    print(f"\nR² = {model.rsquared:.4f}")
    return model


@app.cell
def _(data, model, plot_lm_res, plt):
    """4. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Temp", ylabel="Wing", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()