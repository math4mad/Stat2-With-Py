# /// notebook
# # ch05 · Teen Pregnancy – One-Way ANOVA
#
# State's role in Civil War affects teen pregnancy rates?
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.stats import one_way_anova
    return Path, Stat2Table, load_rda, mo, np, one_way_anova, pl, plt, sns, sys


@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(545, "TeenPregnancy", "state's role in civilwar affects now teen pregnancy?", ["CivilWar", "Teen"])
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    cats = df["CivilWar"].unique().sort().to_list()
    groups = {c: df.filter(pl.col("CivilWar") == c)["Teen"].to_numpy() for c in cats}
    for c, g in groups.items():
        print(f"{c}: n={len(g)}, mean={g.mean():.2f}")
    df.head(8)
    return cats, df, groups


@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df.to_pandas(), x="CivilWar", y="Teen", width=0.4, ax=_ax)
    sns.stripplot(data=df.to_pandas(), x="CivilWar", y="Teen", color="black", alpha=0.5, size=6, ax=_ax)
    plt.gca()
    return


@app.cell
def _(cats, groups, one_way_anova):
    result = one_way_anova(*[groups[c] for c in cats])
    print(f"One-way ANOVA: F={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    print(f"  {'reject H₀' if result['pvalue'] < 0.05 else 'fail to reject H₀'}")
    return result


if __name__ == "__main__":
    app.run()