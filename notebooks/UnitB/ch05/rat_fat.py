# /// notebook
# # ch05 · Rat Fat – One-Way ANOVA
#
# High protein diet: compare weight gain across Beef, Cereal, Pork sources.
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
    """Title."""
    desc = Stat2Table(
        page=540,
        name="FatRats",
        question="high protein diet",
        feature=["Gain", "Protein", "Source"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data, filter high-protein group."""
    df = load_rda(desc.name)
    high_protein = df.filter(pl.col("Protein") == "Hi")
    sources = high_protein["Source"].unique().sort().to_list()
    groups = {s: high_protein.filter(pl.col("Source") == s)["Gain"].to_numpy() for s in sources}
    for s, g in groups.items():
        print(f"{s}: n={len(g)}, mean={g.mean():.2f}")
    high_protein.head(8)
    return df, groups, high_protein, sources


@app.cell
def _(high_protein, sources, plt, sns):
    """2. Dot + boxplot by source."""
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=high_protein.to_pandas(), x="Source", y="Gain", width=0.4, ax=_ax)
    sns.stripplot(data=high_protein.to_pandas(), x="Source", y="Gain", color="black", alpha=0.5, size=6, ax=_ax)
    _ax.set_title("Weight Gain by Protein Source (High Protein)")
    plt.gca()
    return


@app.cell
def _(groups, sources, one_way_anova):
    """3. One-way ANOVA."""
    result = one_way_anova(*[groups[s] for s in sources])
    print("One-way ANOVA")
    print(f"  F = {result['statistic']:.4f}")
    print(f"  p = {result['pvalue']:.4f}")
    print(f"  df = ({result['df_between']}, {result['df_within']})")
    print(f"  {'reject H₀' if result['pvalue'] < 0.05 else 'fail to reject H₀'}")
    return result


if __name__ == "__main__":
    app.run()