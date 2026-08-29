# /// notebook
# # ch02 · Metabolic Rate – Log-Log Regression
#
# Log-log linear regression of metabolic rate on body size.
# Includes ANOVA table and R².
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
    from stat2lib.stats import fit_lm, anova_table
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, pl, plot_lm_res, plot_pair_scatter, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=256,
        name="MetabolicRate",
        question="metabolic-rate-log-transform-reg",
        feature=["LogBodySize", "LogMrate"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(["LogBodySize", "LogMrate"])
    data.head(8)
    return data, df


@app.cell
def _(data, plot_pair_scatter, plt):
    """2. Scatter plot."""
    _fig, _ax = plot_pair_scatter(data, xlabel="LogBodySize", ylabel="LogMrate")
    plt.gca()
    return


@app.cell
def _(data, fit_lm):
    """3. Fit model: LogMrate ~ LogBodySize."""
    model = fit_lm(data, "LogMrate ~ LogBodySize")
    print(model.summary())
    print(f"\nR² = {model.rsquared:.4f}")
    return model


@app.cell
def _(model, anova_table):
    """4. ANOVA table."""
    aov = anova_table(model)
    print("ANOVA Table:")
    print(aov)
    return aov


@app.cell
def _(data, model, plot_lm_res, plt):
    """5. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="LogBodySize", ylabel="LogMrate", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()