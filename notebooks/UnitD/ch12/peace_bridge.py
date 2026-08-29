# /// notebook
# # ch12 · Peace Bridge 2003 – Long Seasonal + ACF
#
# Linear+cosine vs linear+seasonal means. ACF plot of differenced series.
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
    desc = Stat2Table(1421, "PeaceBridge2003", "longer seasonal time series", ["Year", "Month", "Traffic", "t"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl, np):
    df = load_rda(desc.name)
    cost = np.cos(2 * np.pi * df["t"].to_numpy() / 12)
    sint = np.sin(2 * np.pi * df["t"].to_numpy() / 12)
    df = df.with_columns([pl.Series("cost", cost), pl.Series("sint", sint)])
    df.head(8)
    return df

@app.cell
def _(df, fit_lm):
    m1 = fit_lm(df, "Traffic ~ t + cost + sint")
    m2 = fit_lm(df, "Traffic ~ t + C(Month)")
    se1 = np.sqrt(np.mean(m1.resid**2)); se2 = np.sqrt(np.mean(m2.resid**2))
    print(f"Linear+Cos+Sin: R²={m1.rsquared:.4f}, SE={se1:.2f}")
    print(f"Linear+Month:   R²={m2.rsquared:.4f}, SE={se2:.2f}")
    return m1, m2

@app.cell
def _(df, m1, m2, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4))
    _ax1.plot(df["t"], df["Traffic"], "o-", markersize=2, alpha=0.5, label="Data")
    _ax1.plot(df["t"], m1.predict(), "r-", linewidth=1.5, label="Cos+Sin")
    _ax1.set_xlabel("t"); _ax1.set_ylabel("Traffic"); _ax1.legend()
    _ax2.plot(df["t"], m1.resid, "b-", linewidth=1, label="Cos+Sin resid")
    _ax2.plot(df["t"], m2.resid, "orange", linewidth=1, label="Month resid")
    _ax2.axhline(0, color="red", linestyle="--"); _ax2.legend(); _ax2.set_xlabel("t")
    plt.gca(); return

@app.cell
def _(df, np, plt):
    """ACF of differenced series."""
    _traffic = df["Traffic"].to_numpy()
    _diff = _traffic[1:] - _traffic[:-1]
    _n = len(_diff); _acf = [1.0]
    for _k in range(1, min(30, _n)): _acf.append(np.corrcoef(_diff[:_n-_k], _diff[_k:])[0, 1])
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.stem(range(len(_acf)), _acf, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax.set_xlabel("Lag"); _ax.set_ylabel("ACF"); _ax.set_title("ACF of Differenced Traffic")
    _ax.axhline(0, color="black", linewidth=0.5)
    plt.gca(); return

if __name__ == "__main__": app.run()