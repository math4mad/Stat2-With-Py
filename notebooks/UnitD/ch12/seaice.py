# /// notebook
# # ch12 · Sea Ice – Time Series Regression
#
# Arctic sea ice extent over time. Linear, t-scale, and quadratic models.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm
    return Path, Stat2Table, fit_lm, load_rda, mo, np, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1406, "SeaIce", "sea ice time series", ["Year", "Extent", "Area", "t"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda):
    data = load_rda(desc.name); data.head(10)
    return data

@app.cell
def _(data, plt):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(data["Year"], data["Extent"], "o-", markersize=4, color="purple", alpha=0.6)
    _ax.set_xlabel("Year"); _ax.set_ylabel("Extent"); _ax.set_title("Arctic Sea Ice Extent")
    plt.gca(); return

@app.cell
def _(data, fit_lm):
    m1 = fit_lm(data, "Extent ~ Year")
    m2 = fit_lm(data, "Extent ~ t")
    data2 = data.with_columns(pl.col("t").pow(2).alias("t2"))
    m3 = fit_lm(data2, "Extent ~ t + t2")
    print("=== Linear (Year) ===\n", m1.summary())
    print("\n=== Linear (t-scale) ===\n", m2.summary())
    print(f"\nR²(quadratic) = {m3.rsquared:.4f}")
    return m1, m2, m3

@app.cell
def _(data, m1, m2, plt, np):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.plot(data["Year"], data["Extent"], "o", markersize=4, color="purple", alpha=0.6)
    _ax1.plot(data["Year"], m1.predict(), "r-", linewidth=2, label="Linear fit")
    _ax1.set_xlabel("Year"); _ax1.set_ylabel("Extent"); _ax1.legend()
    _ax2.stem(m2.resid, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    _ax2.set_xlabel("t"); _ax2.set_ylabel("Residuals"); _ax2.set_title("t-scale residuals")
    plt.gca(); return

if __name__ == "__main__": app.run()