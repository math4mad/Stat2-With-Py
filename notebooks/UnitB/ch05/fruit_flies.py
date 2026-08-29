# /// notebook
# # ch05 · Fruit Flies – Sexual Activity & Lifespan
#
# One-way ANOVA: Longevity ~ Treatment. Includes residual diagnostics by group.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, anova_table
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, np, pl, plt, sns, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(583, "FruitFlies", "male sexual activity shorten life long?", ["Treatment", "Longevity"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name); df.head(5)
    return df

@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df.to_pandas(), x="Treatment", y="Longevity", width=0.4, ax=_ax)
    sns.stripplot(data=df.to_pandas(), x="Treatment", y="Longevity", color="black", alpha=0.3, size=5, ax=_ax)
    _ax.tick_params(axis="x", rotation=20); plt.gca(); return

@app.cell
def _(df, fit_lm, anova_table):
    model = fit_lm(df, "Longevity ~ Treatment")
    aov = anova_table(model)
    print("ANOVA: Longevity ~ Treatment")
    print(aov)
    return aov, model

@app.cell
def _(df, model, plt, sns):
    """Residuals by group."""
    _res = model.resid
    _pdf = df.to_pandas(); _pdf["Residuals"] = _res
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=_pdf, x="Treatment", y="Residuals", width=0.4, ax=_ax)
    _ax.axhline(0, color="black", linewidth=0.8); _ax.tick_params(axis="x", rotation=20)
    plt.gca(); return

if __name__ == "__main__": app.run()