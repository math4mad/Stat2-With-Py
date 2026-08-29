# /// notebook
# # ch01 · Studentized Residuals – Long Jump Olympics
#
# Olympic long jump gold medal distances over time.
# Detect outliers using studentized (standardised) residuals.
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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import _add_border as add_border
    from stat2lib.stats import fit_lm
    import marimo as mo
    return (
        Path, Stat2Table, add_border, fit_lm, load_rda,
        mo, np, pl, plt, scipy_stats, sys,
    )


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=172,
        name="LongJumpOlympics2016",
        question="Outlier detection with studentized residuals",
        feature=["Year", "Gold"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda):
    """1. Load data."""
    df = load_rda(desc.name)
    df.head(8)
    return df


@app.cell
def _(df, fit_lm):
    """2. Fit linear model."""
    model = fit_lm(df, "Gold ~ Year")
    print(model.summary())
    return model


@app.cell
def _(model, np):
    """3. Studentized residuals."""
    raw_res = model.resid
    z_res = (raw_res - np.mean(raw_res)) / np.std(raw_res, ddof=1)
    influence = model.get_influence()
    stud_res = influence.resid_studentized_external
    print(f"Raw residuals range:     [{raw_res.min():.4f}, {raw_res.max():.4f}]")
    print(f"Z-score residuals range: [{z_res.min():.4f}, {z_res.max():.4f}]")
    print(f"Studentized residuals:   [{stud_res.min():.4f}, {stud_res.max():.4f}]")
    return stud_res


@app.cell
def _(df, plt, stud_res, add_border):
    """4. Plot studentized residuals."""
    _fig, _ax = plt.subplots(figsize=(10, 5))
    add_border(_ax)
    _ax.set_xlabel("Year")
    _ax.set_ylabel("Studentized Residuals")
    _ax.set_title("Long Jump Olympics: Studentized Residuals")
    _ax.set_xlim(1900, 2020)
    _ax.set_ylim(-4, 4)
    _ax.scatter(
        df["Year"], stud_res,
        s=80, c="red", alpha=0.4, edgecolors="purple", linewidths=1.2,
        zorder=3,
    )
    for _y in [-3, -2, 2, 3]:
        _ax.axhline(_y, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    _ax.axhspan(-2, 2, alpha=0.05, color="green", label="±2 band")
    _ax.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()