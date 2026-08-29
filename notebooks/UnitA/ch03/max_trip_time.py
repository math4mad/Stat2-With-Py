# /// notebook
# # ch03 · Funnel Drop – Response Surface
#
# Full quadratic model with interaction: Time ~ Funnel*Tube + Funnel² + Tube².
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
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=346,
        name="FunnelDrop",
        question="maximize time",
        feature=["Funnel", "Tube", "Time"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and add squared + interaction terms."""
    df = load_rda(desc.name)
    data = df.with_columns([
        pl.col("Funnel").pow(2).alias("Funnel2"),
        pl.col("Tube").pow(2).alias("Tube2"),
        (pl.col("Funnel") * pl.col("Tube")).alias("Funnel_Tube"),
    ])
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Fit model: Time ~ Funnel + Tube + Funnel² + Tube² + Funnel:Tube."""
    model = fit_lm(data, "Time ~ Funnel + Tube + Funnel2 + Tube2 + Funnel_Tube")
    print(model.summary())
    return model


@app.cell
def _(data, model, plt):
    """3. Residual diagnostics."""
    _res = model.resid
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.stem(_res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax1.axhline(0, color="black", linewidth=0.8)
    _ax1.set_xlabel("Index")
    _ax1.set_ylabel("Residuals")
    import scipy.stats as stats
    stats.probplot(_res, dist="norm", plot=_ax2)
    _ax2.get_lines()[0].set_markerfacecolor("purple")
    _ax2.get_lines()[0].set_alpha(0.6)
    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()