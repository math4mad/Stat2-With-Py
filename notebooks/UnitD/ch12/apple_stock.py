# /// notebook
# # ch12 · Apple Stock – Random Walk & ACF
#
# Stock price as random walk. Differencing + ACF to detect autocorrelation.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    return Path, Stat2Table, load_rda, mo, np, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1431, "AppleStock", "stock time series", ["Date", "Price", "Change", "Volume"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, np):
    df = load_rda(desc.name)
    price = df["Price"].to_numpy()
    np.random.seed(34343)
    rw = [np.cumsum(np.concatenate([[100], np.random.normal(0, 1.5, 65)])) for _ in range(3)]
    df.head(5)
    return df, price, rw

@app.cell
def _(price, rw, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    _series = [price] + rw
    for i, (_ax, _s) in enumerate(zip(_axes.flat, _series)):
        _ax.plot(_s, linewidth=1); _ax.set_title(f"Series {i+1}" if i>0 else "Apple Stock Price")
    plt.gca(); return

@app.cell
def _(price, np, plt):
    """Differencing and ACF."""
    _diff = price[1:] - price[:-1]
    _n = len(_diff); _acf = [1.0]
    for _k in range(1, min(25, _n)): _acf.append(np.corrcoef(_diff[:_n-_k], _diff[_k:])[0, 1])
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.plot(price[1:], price[:-1], "o", markersize=3, alpha=0.5, color="purple")
    _ax1.set_xlabel("Previous Close"); _ax1.set_ylabel("Price")
    _ax2.stem(_acf, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax2.set_xlabel("Lag"); _ax2.set_ylabel("ACF"); _ax2.set_title("ACF of Differenced Price")
    _ax2.axhline(0, color="black", linewidth=0.5)
    plt.gca(); return

if __name__ == "__main__": app.run()