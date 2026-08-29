# /// notebook
# # ch06 · Frantic Fingers – Two-Way ANOVA
#
# Drug effect on response ability. One-way + two-way ANOVA: Rate ~ Drug + Subj.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, anova_table, one_way_anova
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, np, one_way_anova, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(699, "FranticFingers", "drug affect response ability", ["ID", "Rate", "Subj", "Drug"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    df.head(8)
    return df

@app.cell
def _(df, plt):
    """Side-by-side line plot by subject."""
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _s in df["Subj"].unique().to_list():
        _d = df.filter(pl.col("Subj") == _s).sort("Drug")
        _ax.plot(["Pi", "Ca", "H"], _d["Rate"], "o-", label=str(_s))
    _ax.legend(); _ax.set_ylabel("Rate"); plt.gca(); return

@app.cell
def _(df, one_way_anova):
    """One-way ANOVA by Drug."""
    drugs = df["Drug"].unique().sort().to_list()
    groups = [df.filter(pl.col("Drug") == d)["Rate"].to_numpy() for d in drugs]
    result = one_way_anova(*groups)
    print(f"One-way ANOVA (Drug): F={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    return result

@app.cell
def _(df, fit_lm, anova_table):
    """Two-way ANOVA: Rate ~ Drug + Subj."""
    model = fit_lm(df, "Rate ~ Drug + Subj")
    aov = anova_table(model)
    print("Two-way ANOVA: Rate ~ Drug + Subj")
    print(aov)
    return aov, model

if __name__ == "__main__": app.run()