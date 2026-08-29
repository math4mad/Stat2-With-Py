# /// notebook
# # ch04 · One-Hot Encoding – Three Cars 2017
#
# Data exploration with categorical variable CarType.
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
    return Path, Stat2Table, load_rda, mo, pl, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=485,
        name="ThreeCars2017",
        question="one hot encoding",
        feature=["CarType", "Age", "Price", "Mileage", "Mazda6", "Accord", "Maxima"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and explore."""
    df = load_rda(desc.name)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"\nCarType values: {df['CarType'].unique().to_list()}")
    df.head(10)
    return df


@app.cell
def _(df, pl):
    """2. One-hot encode CarType."""
    dummies = df.select(["CarType"]).to_dummies("CarType")
    result = pl.concat([df, dummies], how="horizontal")
    result.head(8)
    return dummies, result


if __name__ == "__main__":
    app.run()