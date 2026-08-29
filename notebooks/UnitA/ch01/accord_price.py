# /// notebook
# # ch01 · p106 – Accord Price Regression
#
# Used Honda Accord prices regressed on mileage.
# Includes boxplots by mileage group, scatter + fit, residuals,
# Q-Q plot, and density plot.
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
    import scipy.stats as stats
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import _add_border as add_border
    from stat2lib.stats import fit_lm
    import marimo as mo
    return Path, Stat2Table, add_border, fit_lm, load_rda, mo, np, pl, plt, stats, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=105,
        name="AccordPrice",
        question="Used Accord price vs mileage",
        feature=["Age", "Price", "Mileage"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(["Mileage", "Price"])
    data.head(8)
    return data, df


@app.cell
def _(df, pl):
    """2. Data exploration."""
    low_group = df.filter(pl.col("Mileage") <= 25.0)
    high_group = df.filter(pl.col("Mileage") >= 65.0)
    mid_group = df.filter(
        (pl.col("Mileage") > 25.0) & (pl.col("Mileage") < 65.0)
    )
    groups = [low_group, mid_group, high_group]
    medians = [g["Mileage"].median() for g in groups]
    corr_val = df.select(pl.corr("Mileage", "Price")).item()
    print(f"Correlation (Mileage, Price): {corr_val:.4f}")
    print(f"Median mileage by group: {[round(m, 0) for m in medians]}")
    return corr_val, groups, high_group, low_group, medians, mid_group


@app.cell
def _(groups, plt, add_border):
    """3. Boxplot by mileage group."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _ax.set_xlabel("Mileage group")
    _ax.set_ylabel("Price")
    _ax.set_title("Accord Price by Mileage Group")
    _group_labels = ["≤ 25k", "25k–65k", "≥ 65k"]
    _positions = [1, 2, 3]
    for _pos, _grp in zip(_positions, groups):
        _prices = _grp["Price"].to_numpy()
        _bp = _ax.boxplot(_prices, positions=[_pos], widths=0.5, patch_artist=True)
        _bp["boxes"][0].set_facecolor("lightblue")
        _bp["boxes"][0].set_alpha(0.6)
    _ax.set_xticks(_positions)
    _ax.set_xticklabels(_group_labels)
    plt.gca()
    return


@app.cell
def _(df, plt, add_border):
    """4. Scatter plot."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _ax.set_xlabel("Mileage")
    _ax.set_ylabel("Price")
    _ax.set_title("Accord Price vs Mileage")
    _ax.scatter(
        df["Mileage"], df["Price"],
        s=80, c="lightgreen", alpha=0.5, edgecolors="black", linewidths=0.8,
    )
    plt.gca()
    return


@app.cell
def _(data, fit_lm):
    """5. Linear regression."""
    model = fit_lm(data, "Price ~ Mileage")
    print(model.summary())
    print(f"\nR² = {model.rsquared:.4f}")
    return model


@app.cell
def _(df, model, plt, add_border):
    """6. Fitted line."""
    _xs = df["Mileage"].to_numpy()
    _y_hat = model.predict()
    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _ax.set_xlabel("Mileage")
    _ax.set_ylabel("Price")
    _ax.set_title("Accord Price: Regression Line")
    _ax.scatter(
        _xs, df["Price"],
        s=80, c="lightgreen", alpha=0.5, edgecolors="black", linewidths=0.8,
    )
    _ax.plot(_xs, _y_hat, color="blue", linewidth=2, label="Fit line")
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(model, np, plt, stats, add_border):
    """7. Residual diagnostics."""
    _res = model.resid
    _y_hat_pred = model.predict()

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 10))
    (_ax1, _ax2), (_ax3, _ax4) = _axes
    _fig.suptitle("Accord Price Regression Diagnostics")

    for _a in (_ax1, _ax2, _ax3, _ax4):
        add_border(_a)

    _ax1.stem(_res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    _ax1.axhline(0, color="black", linewidth=0.8)
    _ax1.set_xlabel("Index")
    _ax1.set_ylabel("Residuals")

    _ax2.scatter(
        _y_hat_pred, _res, s=60, c="purple", alpha=0.4,
        edgecolors="black", linewidths=0.8,
    )
    _ax2.axhline(0, color="black", linewidth=0.8)
    _ax2.set_xlabel("Fitted values")
    _ax2.set_ylabel("Residuals")

    stats.probplot(_res, dist="norm", plot=_ax3)
    _ax3.get_lines()[0].set_markerfacecolor("purple")
    _ax3.get_lines()[0].set_markeredgecolor("black")
    _ax3.get_lines()[0].set_alpha(0.6)
    _ax3.set_xlabel("Theoretical quantiles")
    _ax3.set_ylabel("Sample quantiles")

    _ax4.hist(
        _res, bins=12, density=True, color="lightblue", alpha=0.6,
        edgecolor="black",
    )
    from scipy.stats import gaussian_kde
    _kde = gaussian_kde(_res)
    _x_range = np.linspace(_res.min(), _res.max(), 200)
    _ax4.plot(_x_range, _kde(_x_range), color="blue", linewidth=2, label="Density")
    _ax4.set_xlabel("Residuals")
    _ax4.set_ylabel("Density")
    _ax4.legend()

    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()