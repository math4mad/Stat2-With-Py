# /// notebook
# # ch03 · Fish – Nonlinear Length×Width Feature
#
# Compare simple model (Weight ~ Length×Width) vs full model
# (Weight ~ Length + Width + Length×Width).
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
        page=325,
        name="Perch",
        question="fish-length-width-weight-predict",
        feature=["Length", "Width", "Weight"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and create cross-feature."""
    df = load_rda(desc.name)
    data = df.with_columns((pl.col("Length") * pl.col("Width")).alias("lengthxwidth"))
    data.head(8)
    return data


@app.cell
def _(data, fit_lm):
    """2. Model 1: Weight ~ lengthxwidth."""
    model1 = fit_lm(data, "Weight ~ lengthxwidth")
    print("=== Model 1: Weight ~ lengthxwidth ===")
    print(model1.summary())
    print(f"R² = {model1.rsquared:.4f}")
    return model1


@app.cell
def _(data, fit_lm):
    """3. Model 2: Weight ~ Length + Width + lengthxwidth."""
    model2 = fit_lm(data, "Weight ~ Length + Width + lengthxwidth")
    print("=== Model 2: Weight ~ Length + Width + lengthxwidth ===")
    print(model2.summary())
    print(f"R² = {model2.rsquared:.4f}")
    return model2


if __name__ == "__main__":
    app.run()