# /// notebook
# # ch01 · Example 1.8 – Species–Area Relationship
#
# Number of mammal species on islands vs island area.
# Log-transforms linearise the power-law relationship.
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
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import _add_border as add_border
    from stat2lib.stats import fit_lm
    import marimo as mo
    return Path, Stat2Table, add_border, fit_lm, load_rda, mo, np, pl, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=146,
        name="SpeciesArea",
        question="Number of mammal species related to island area",
        feature=["Area", "Species", "logArea", "logSpecies"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data1 = df.select(["logArea", "Species"])
    data2 = df.select(["logArea", "logSpecies"])
    df.head(8)
    return data1, data2, df


@app.cell
def _(data2, plt, add_border):
    """2. Scatter plot (log–log)."""
    _fig, _ax = plt.subplots(figsize=(7, 5))
    add_border(_ax)
    _ax.set_xlabel("log(Area)")
    _ax.set_ylabel("log(Species)")
    _ax.scatter(
        data2["logArea"], data2["logSpecies"],
        s=80, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8,
    )
    _ax.set_title("log(Species) vs log(Area)")
    plt.gca()
    return


@app.cell
def _(data1, fit_lm):
    """3. Model 1: Species ~ log(Area)."""
    model1 = fit_lm(data1, "Species ~ logArea")
    print("=== Model 1: Species ~ log(Area) ===")
    print(model1.summary())
    return model1


@app.cell
def _(data2, fit_lm):
    """4. Model 2: log(Species) ~ log(Area)."""
    model2 = fit_lm(data2, "logSpecies ~ logArea")
    print("=== Model 2: log(Species) ~ log(Area) ===")
    print(model2.summary())
    return model2


@app.cell
def _(data1, data2, model1, model2, plt, add_border):
    """5. Side-by-side regression plots."""
    _y_hat1 = model1.predict()
    _y_hat2 = model2.predict()

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _a in (_ax1, _ax2):
        add_border(_a)

    _ax1.set_xlabel("log(Area)")
    _ax1.set_ylabel("Species")
    _ax1.scatter(
        data1["logArea"], data1["Species"],
        s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8,
    )
    _ax1.plot(data1["logArea"], _y_hat1, color="blue", linewidth=2, label="Fit")
    _ax1.legend()
    _ax1.set_title("Species ~ log(Area)")

    _ax2.set_xlabel("log(Area)")
    _ax2.set_ylabel("log(Species)")
    _ax2.scatter(
        data2["logArea"], data2["logSpecies"],
        s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8,
    )
    _ax2.plot(data2["logArea"], _y_hat2, color="blue", linewidth=2, label="Fit")
    _ax2.legend()
    _ax2.set_title("log(Species) ~ log(Area)")

    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()