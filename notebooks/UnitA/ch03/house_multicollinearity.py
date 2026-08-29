# /// notebook
# # ch03 · Houses NY – Multicollinearity Detection
#
# Correlation matrix + nested model comparison for Price ~ Beds + Baths + Size + Lot.
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
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.stats import fit_lm, correlation_matrix
    return Path, Stat2Table, correlation_matrix, fit_lm, load_rda, mo, np, pl, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=357,
        name="HousesNY",
        question="detecting multicollinearity",
        feature=["Beds", "Baths", "Size", "Lot", "Price"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data = df.select(desc.feature)
    data.head(8)
    return data, df


@app.cell
def _(data, correlation_matrix):
    """2. Correlation matrix."""
    corr = correlation_matrix(data, data.columns)
    print("Correlation Matrix:")
    print(corr)
    return corr


@app.cell
def _(data, fit_lm):
    """3. Full model: Price ~ Beds + Baths + Size + Lot."""
    model = fit_lm(data, "Price ~ Beds + Baths + Size + Lot")
    print(model.summary())
    return model


@app.cell
def _(data, fit_lm):
    """4. Nested models – incremental R²."""
    import statsmodels.formula.api as smf
    pdf = data.to_pandas()
    formulas = [
        "Price ~ 1",
        "Price ~ 1 + Baths",
        "Price ~ 1 + Baths + Beds",
        "Price ~ 1 + Baths + Beds + Size",
        "Price ~ 1 + Baths + Beds + Size + Lot",
    ]
    for i, f in enumerate(formulas):
        m = smf.ols(f, data=pdf).fit()
        print(f"Model {i+1}: {f:45s}  R²={m.rsquared:.4f}")
    return


if __name__ == "__main__":
    app.run()