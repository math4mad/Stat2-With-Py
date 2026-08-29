# /// notebook
# # ch03 · Hospital Beds – Interaction Model
#
# sqrt(MDs) ~ Hospitals + Beds + Hospitals:Beds with VIF check.
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
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.stats import fit_lm
    return Path, Stat2Table, fit_lm, load_rda, mo, pl, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=361,
        name="CountyHealth",
        question="sqrt transform of MDs with interaction",
        feature=["Hospitals", "Beds", "sqrtMDs"],
    )
    mo.md(f"## {desc.question} (interaction)")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data, sqrt-transform, add interaction."""
    df = load_rda(desc.name)
    data = df.with_columns([
        pl.col("MDs").sqrt().alias("sqrtMDs"),
        (pl.col("Hospitals") * pl.col("Beds")).alias("Hosp_Beds"),
    ])
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Fit model: sqrtMDs ~ Hospitals + Beds + Hospitals:Beds."""
    model = fit_lm(data, "sqrtMDs ~ Hospitals + Beds + Hosp_Beds")
    print(model.summary())
    return model


@app.cell
def _(data):
    """3. VIF (Variance Inflation Factor)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd
    import numpy as np
    _pdf = data.select(["Hospitals", "Beds", "Hosp_Beds"]).to_pandas()
    _vif = pd.DataFrame({
        "feature": ["Hospitals", "Beds", "Hosp_Beds"],
        "VIF": [variance_inflation_factor(_pdf.values, i) for i in range(_pdf.shape[1])],
    })
    print("VIF:")
    print(_vif)
    return


if __name__ == "__main__":
    app.run()