# /// notebook
# # ch03 · Customer Spending – Feature Engineering
#
# Predict Amount from Dollar12, then engineered feature AvgSpent12 = Dollar12/Freq12.
# Includes nested model comparison and quadratic term.
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
        page=380,
        name="Clothing",
        question="prediction customer spending",
        feature=["Recency", "Freq12", "Dollar12", "Freq24", "Dollar24", "Card", "Amount"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data and filter outliers."""
    df = load_rda(desc.name)
    data = df.filter((pl.col("Amount") > 0) & (pl.col("Amount") < 1_506_000))
    print(f"Filtered: {df.shape[0]} → {data.shape[0]} rows")
    data.head(8)
    return data, df


@app.cell
def _(data, fit_lm):
    """2. Simple model: Amount ~ Dollar12."""
    model1 = fit_lm(data, "Amount ~ Dollar12")
    print("=== Amount ~ Dollar12 ===")
    print(model1.summary())
    print(f"R² = {model1.rsquared:.4f}")
    return model1


@app.cell
def _(data, pl):
    """3. Engineer feature: AvgSpent12 = Dollar12 / Freq12."""
    complex_data = data.filter(pl.col("Dollar12") > 0).with_columns(
        (pl.col("Dollar12") / pl.col("Freq12")).alias("AvgSpent12")
    )
    print(f"Rows with Dollar12 > 0: {complex_data.shape[0]}")
    complex_data.head(5)
    return complex_data


@app.cell
def _(complex_data, fit_lm):
    """4. Model: Amount ~ AvgSpent12."""
    model2 = fit_lm(complex_data, "Amount ~ AvgSpent12")
    print("=== Amount ~ AvgSpent12 ===")
    print(model2.summary())
    print(f"R² = {model2.rsquared:.4f}")
    return model2


@app.cell
def _(complex_data, fit_lm, pl):
    """5. Quadratic model: Amount ~ AvgSpent12 + AvgSpent12²."""
    cdata = complex_data.with_columns(pl.col("AvgSpent12").pow(2).alias("AvgSpent12_2"))
    model3 = fit_lm(cdata, "Amount ~ AvgSpent12 + AvgSpent12_2")
    print("=== Amount ~ AvgSpent12 + AvgSpent12² ===")
    print(model3.summary())
    print(f"R² = {model3.rsquared:.4f}")
    return model3


if __name__ == "__main__":
    app.run()