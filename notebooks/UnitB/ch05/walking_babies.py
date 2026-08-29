# /// notebook
# # ch05 · Walking Babies – One-Way ANOVA (LM approach)
#
# Does exercise reduce first walking time? ANOVA via LM: Age ~ Group.
# Includes Levene's test for homogeneity of variance.
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
    from stat2lib.stats import fit_lm, anova_table, one_way_anova, levene_test
    return Path, Stat2Table, anova_table, fit_lm, levene_test, load_rda, mo, np, one_way_anova, pl, plt, sns, sys


@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(552, "WalkingBabies", "exercise reduce first walking time of baby?", ["Group", "Age"])
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    cats = df["Group"].unique().sort().to_list()
    groups = {c: df.filter(pl.col("Group") == c)["Age"].to_numpy() for c in cats}
    for c, g in groups.items():
        print(f"{c}: n={len(g)}, mean={g.mean():.2f}")
    df.head(8)
    return cats, df, groups


@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df.to_pandas(), x="Group", y="Age", width=0.4, ax=_ax)
    sns.stripplot(data=df.to_pandas(), x="Group", y="Age", color="black", alpha=0.5, size=6, ax=_ax)
    _ax.tick_params(axis="x", rotation=20)
    plt.gca()
    return


@app.cell
def _(cats, groups, one_way_anova):
    result = one_way_anova(*[groups[c] for c in cats])
    print(f"One-way ANOVA: F={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    return result


@app.cell
def _(df, fit_lm, anova_table):
    """ANOVA via LM: Age ~ Group."""
    model = fit_lm(df, "Age ~ Group")
    aov = anova_table(model)
    print("ANOVA via LM:")
    print(aov)
    return aov, model


@app.cell
def _(cats, groups, levene_test):
    result = levene_test(*[groups[c] for c in cats])
    print(f"Levene's test: W={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    return result


if __name__ == "__main__":
    app.run()