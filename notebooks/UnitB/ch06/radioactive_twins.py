# /// notebook
# # ch06 · Radioactive Twins – Paired Comparison
#
# Lung clearance ability: city vs rural (paired twins).
# One-way ANOVA on paired data.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import one_way_anova
    return Path, Stat2Table, load_rda, mo, np, one_way_anova, pl, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(690, "RadioactiveTwins", "living on city or rural, lung clean ability diff?", ["TwinPair", "Env", "Rate"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    envs = df["Env"].unique().sort().to_list()
    groups = {e: df.filter(pl.col("Env") == e)["Rate"].to_numpy() for e in envs}
    for e, g in groups.items(): print(f"{e}: n={len(g)}, mean={g.mean():.2f}")
    df.head(8)
    return df, envs, groups

@app.cell
def _(envs, groups, one_way_anova):
    result = one_way_anova(*[groups[e] for e in envs])
    print(f"One-way ANOVA: F={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    print(f"  {'reject H₀' if result['pvalue'] < 0.05 else 'fail to reject H₀'}")
    return result

if __name__ == "__main__": app.run()