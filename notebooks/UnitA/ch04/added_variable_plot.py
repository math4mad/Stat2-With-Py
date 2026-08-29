# /// notebook
# # ch04 · Added Variable Plot – Houses NY
#
# Demonstrate added-variable plot: regress residuals of Price~Beds
# on residuals of Size~Beds.
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
        page=435,
        name="HousesNY",
        question="added variable plot",
        feature=["Beds", "Size", "Price"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(["Beds", "Size", "Price"])
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Price ~ Beds."""
    model_price_beds = fit_lm(data, "Price ~ Beds")
    print("=== Price ~ Beds ===")
    print(model_price_beds.summary())
    print(f"R² = {model_price_beds.rsquared:.4f}")
    return model_price_beds


@app.cell
def _(data, fit_lm):
    """3. Size ~ Beds."""
    model_size_beds = fit_lm(data, "Size ~ Beds")
    print("=== Size ~ Beds ===")
    print(model_size_beds.summary())
    print(f"R² = {model_size_beds.rsquared:.4f}")
    return model_size_beds


@app.cell
def _(model_price_beds, model_size_beds, plt, add_border, np):
    """4. Added-variable plot: residuals(Price~Beds) vs residuals(Size~Beds)."""
    _res_price = model_price_beds.resid
    _res_size = model_size_beds.resid

    _fig, _ax = plt.subplots(figsize=(8, 5))
    add_border(_ax)
    _ax.scatter(_res_size, _res_price, s=60, c="purple", alpha=0.4,
                edgecolors="black", linewidths=0.8)
    _ax.set_xlabel("Residuals: Size ~ Beds")
    _ax.set_ylabel("Residuals: Price ~ Beds")
    _ax.set_title("Added Variable Plot")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()