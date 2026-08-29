# /// notebook
# # ch03 · Brain pH – Gender Differences
#
# Grouped regression: pH ~ Age for Male and Female separately.
# Combined plot with fit lines.
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
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import _add_border as add_border
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, add_border, fit_lm, load_rda, mo, np, pl, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=409,
        name="BrainpH",
        question="research in brain tissue",
        feature=["pH", "Sex", "Age"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and split by sex."""
    df = load_rda(desc.name)
    sexes = df["Sex"].unique().sort().to_list()
    gdf = {s: df.filter(pl.col("Sex") == s) for s in sexes}
    for _s, _g in gdf.items():
        print(f"{_s}: n={_g.shape[0]}")
    df.head(5)
    return df, gdf, sexes


@app.cell
def _(gdf, sexes, fit_lm):
    """2. Fit separate models."""
    models = {}
    for _s in sexes:
        models[_s] = fit_lm(gdf[_s], "pH ~ Age")
        print(f"=== {_s} ===")
        print(models[_s].summary())
        print()
    return models


@app.cell
def _(gdf, sexes, models, plt, add_border, np):
    """3. Combined plot."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _colors = {"M": "blue", "F": "purple"}
    for _s in sexes:
        _g = gdf[_s]
        _x = _g["Age"].to_numpy()
        _y = _g["pH"].to_numpy()
        _ax.scatter(_x, _y, s=60, c=_colors.get(_s, "gray"), alpha=0.4,
                     edgecolors="black", linewidths=0.8, label=_s)
        _x_sort = np.sort(_x)
        _y_hat = models[_s].predict({"Age": _x_sort})
        _ax.plot(_x_sort, _y_hat, color=_colors.get(_s, "gray"), linewidth=2)
    _ax.set_xlabel("Age")
    _ax.set_ylabel("pH")
    _ax.legend()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()