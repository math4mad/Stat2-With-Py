# /// notebook
# # ch03 · Handwriting – Gender Analysis
#
# Grouped regression: Survey2 ~ Survey1 by Gender.
# Includes indicator variable model.
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
        page=414,
        name="Handwriting",
        question="Handwriting analysis",
        feature=["Gender", "Survey1", "Survey2"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and split by gender."""
    df = load_rda(desc.name)
    genders = df["Gender"].unique().sort().to_list()
    gdf = {g: df.filter(pl.col("Gender") == g) for g in genders}
    for g, d in gdf.items():
        print(f"{g}: n={d.shape[0]}")
    df.head(5)
    return df, gdf, genders


@app.cell
def _(gdf, genders, fit_lm):
    """2. Fit separate models: Survey2 ~ Survey1."""
    models = {}
    for g in genders:
        models[g] = fit_lm(gdf[g], "Survey2 ~ Survey1")
        print(f"=== {g} ===")
        print(models[g].summary())
        print()
    return models


@app.cell
def _(gdf, genders, models, plt, add_border, np):
    """3. Combined plot."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _colors = {genders[0]: "blue", genders[1]: "purple"} if len(genders) >= 2 else {}
    for g in genders:
        _d = gdf[g]
        _x = _d["Survey1"].to_numpy()
        _y = _d["Survey2"].to_numpy()
        _ax.scatter(_x, _y, s=60, c=_colors.get(g, "gray"), alpha=0.4,
                     edgecolors="black", linewidths=0.8, label=str(g))
        _x_sort = np.sort(_x)
        _y_hat = models[g].predict({"Survey1": _x_sort})
        _ax.plot(_x_sort, _y_hat, color=_colors.get(g, "gray"), linewidth=2)
    _ax.set_xlabel("Survey1")
    _ax.set_ylabel("Survey2")
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(df, fit_lm):
    """4. Model with indicator: Survey2 ~ Survey1 + Gender."""
    model = fit_lm(df, "Survey2 ~ Survey1 + Gender")
    print("=== Survey2 ~ Survey1 + Gender ===")
    print(model.summary())
    return model


if __name__ == "__main__":
    app.run()