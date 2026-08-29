# /// notebook
# # ch06 · Sleeping Shrews – One-Way ANOVA + Wide Table
#
# Compare sleep phase rates. Includes long-to-wide pivot.
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
    desc = Stat2Table(693, "SleepingShrews", "sleep status", ["Shrew", "Phase", "Rate"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    phases = df["Phase"].unique().sort().to_list()
    groups = {p: df.filter(pl.col("Phase") == p)["Rate"].to_numpy() for p in phases}
    for p, g in groups.items(): print(f"{p}: n={len(g)}, mean={g.mean():.2f}")
    df.head(8)
    return df, groups, phases

@app.cell
def _(phases, groups, one_way_anova):
    result = one_way_anova(*[groups[p] for p in phases])
    print(f"One-way ANOVA: F={result['statistic']:.4f}, p={result['pvalue']:.4f}")
    return result

@app.cell
def _(df, pl):
    """Pivot to wide table."""
    wide = df.pivot(values="Rate", index="Shrew", columns="Phase")
    print("Wide table (Shrew × Phase):")
    wide
    return wide

if __name__ == "__main__": app.run()