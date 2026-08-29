# /// notebook
# # ch12 · Seasonal Timeseries – Peace Bridge 2012
#
# Cos/sin seasonal model + month dummy model for traffic.
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
    desc = Stat2Table(1412, "PeaceBridge2012", "seasonal time series", ["Year", "Month", "Traffic", "t"])
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
def _(df, plt):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax.plot(df["t"], df["Traffic"], "o-", markersize=3, linewidth=2, color="purple", alpha=0.6)
    _ax.set_xlabel("t (months)"); _ax.set_ylabel("Traffic (1000s)")
    plt.gca(); return

@app.cell
def _(df, fit_lm):
    m1 = fit_lm(df, "Traffic ~ cost + sint")
    m2 = fit_lm(df, "Traffic ~ C(Month)")
    print("=== Cosine + Sine ===")
    print(m1.summary())
    print(f"\nR²(cost+sint) = {m1.rsquared:.4f}")
    print(f"\nR²(month dummy) = {m2.rsquared:.4f}")
    return m1, m2

@app.cell
def _(df, m1, m2, plt):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4))
    _ax1.plot(df["t"], df["Traffic"], "o-", markersize=3, alpha=0.5, label="Data")
    _ax1.plot(df["t"], m1.predict(), "r-", linewidth=2, label="Cos+Sin fit")
    _ax1.set_xlabel("t"); _ax1.set_ylabel("Traffic"); _ax1.legend()
    _ax2.plot(df["t"], m1.resid, "b-", linewidth=1, label="Cos+Sin resid")
    _ax2.plot(df["t"], m2.resid, "orange", linewidth=1, label="Month resid")
    _ax2.axhline(0, color="red", linestyle="--"); _ax2.legend(); _ax2.set_xlabel("t"); _ax2.set_ylabel("Residuals")
    plt.gca(); return

if __name__ == "__main__": app.run()