# /// notebook
# # ch03 · CO₂ – Quadratic Regression
#
# Polynomial regression: CO2 ~ Day + Day².
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
    from stat2lib.plotting import _add_border as add_border, plot_lm_res
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, add_border, fit_lm, load_rda, mo, np, pl, plot_lm_res, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=337,
        name="CO2Germany",
        question="co2-multinomial-reg",
        feature=["Day", "CO2"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.with_columns(pl.col("Day").pow(2).alias("Day2"))
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Fit model: CO2 ~ Day + Day²."""
    model = fit_lm(data, "CO2 ~ Day + Day2")
    print(model.summary())
    return model


@app.cell
def _(data, model, plt, add_border, np):
    """3. Quadratic fit plot."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _x = data["Day"].to_numpy()
    _y = data["CO2"].to_numpy()
    _ax.scatter(_x, _y, s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8)
    _x_sort = np.sort(_x)
    _y_hat = model.predict({"Day": _x_sort, "Day2": _x_sort**2})
    _ax.plot(_x_sort, _y_hat, color="orange", linewidth=3, label="Quadratic fit")
    _ax.set_xlabel("Day")
    _ax.set_ylabel("CO₂")
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(data, model, plot_lm_res, plt):
    """4. Diagnostic plots."""
    _fig = plot_lm_res(data=data, xlabel="Day", ylabel="CO2", model=model)
    plt.gca()
    return


if __name__ == "__main__":
    app.run()