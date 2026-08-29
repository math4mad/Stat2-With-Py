# /// notebook
# # ch01 · Capacitor Voltage Log-Transform
#
# Explore the relationship between capacitor discharge voltage and time.
# A log-transform of voltage linearises the exponential decay.
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
    from stat2lib.plotting import plot_voltage, _add_border as add_border
    from stat2lib.stats import fit_lm
    import marimo as mo
    return (
        Path, Stat2Table, add_border, fit_lm, load_rda,
        mo, np, pl, plot_voltage, plt, sys,
    )


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=182,
        name="Volts",
        question="Capacitor discharge voltage vs time, log-transform effect",
        feature=["Voltage", "Time"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and log-transform."""
    df = load_rda(desc.name)
    df = df.with_columns(pl.col("Voltage").log().alias("logVoltage"))
    df.head(8)
    return df


@app.cell
def _(df, plot_voltage, plt):
    """2. Side-by-side scatter."""
    _fig = plot_voltage(df)
    plt.gca()
    return


@app.cell
def _(df, fit_lm):
    """3. Linear regression: log(Voltage) ~ Time."""
    model = fit_lm(df, "logVoltage ~ Time")
    print(model.summary())
    return model


@app.cell
def _(model, plt, add_border):
    """4. Residual plot."""
    _res = model.resid
    _fig, _ax = plt.subplots(figsize=(8, 4))
    add_border(_ax)
    _ax.stem(_res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("Index")
    _ax.set_ylabel("Residuals")
    _ax.set_title("Residuals: log(Voltage) ~ Time")
    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()