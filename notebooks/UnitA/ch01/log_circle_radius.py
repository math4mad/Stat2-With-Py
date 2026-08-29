# /// notebook
# # ch01 · Circle Area Log-Transform Demo
#
# Demonstrate how a log–log transform linearises the quadratic
# relationship between radius and area of a circle.
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from stat2lib.plotting import _add_border as add_border
    import marimo as mo
    return Path, add_border, mo, np, pl, plt, sys


@app.cell
def _(mo):
    """Title."""
    mo.md("# Circle Area Log-Transform")
    return


@app.cell
def _(np, pl):
    """1. Generate data."""
    rad = np.arange(1, 51, 3)
    area = rad**2 * np.pi
    df = pl.DataFrame({
        "Radius": rad,
        "Area": area,
        "logRadius": np.log(rad),
        "logArea": np.log(area),
    })
    df.head(8)
    return area, df, rad


@app.cell
def _(mo):
    """Title."""
    mo.md("# Circle Area Log-Transform")
    return


@app.cell
def _(area, np, plt, rad, add_border):
    """2. Side-by-side scatter."""
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _a in (_ax1, _ax2):
        add_border(_a)
    _ax1.set_xlabel("Radius")
    _ax1.set_ylabel("Area")
    _ax1.scatter(
        rad, area, s=60, c="purple", alpha=0.4,
        edgecolors="black", linewidths=0.8,
    )
    _ax1.set_title("Area = π·r²  (quadratic)")
    _ax2.set_xlabel("log(Radius)")
    _ax2.set_ylabel("log(Area)")
    _ax2.scatter(
        np.log(rad), np.log(area),
        s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8,
    )
    _ax2.set_title("log(Area) = 2·log(r) + log(π)  (linear)")
    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()