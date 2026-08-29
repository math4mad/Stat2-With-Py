# /// notebook
# # ch05 · Metro Commutes – Levene's Test
#
# Test homogeneity of variance across cities before ANOVA.
# Levene's test rejects H₀ → cannot use standard ANOVA.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import levene_test
    return Path, Stat2Table, levene_test, load_rda, mo, np, pl, plt, sns, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(554, "MetroCommutes", "commute time difference?", ["City", "Distance", "Time"])
    mo.md(f"# {desc.question}")
    return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name); cats = df["City"].unique().sort().to_list()
    groups = {c: df.filter(pl.col("City") == c)["Time"].to_numpy() for c in cats}
    for c, g in groups.items(): print(f"{c}: n={len(g)}, mean={g.mean():.1f}, var={g.var():.1f}")
    df.head(8)
    return cats, df, groups

@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df.to_pandas(), x="City", y="Time", width=0.4, ax=_ax)
    sns.stripplot(data=df.to_pandas(), x="City", y="Time", color="black", alpha=0.5, size=6, ax=_ax)
    plt.gca(); return

@app.cell
def _(cats, groups, levene_test):
    result = levene_test(*[groups[c] for c in cats])
    print(f"Levene's test: W={result['statistic']:.4f}, p={result['pvalue']:.6f}")
    print(f"  {'reject H₀ (variances differ)' if result['pvalue'] < 0.05 else 'fail to reject H₀'}")
    return result

if __name__ == "__main__": app.run()