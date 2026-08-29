# /// notebook
# # ch07 · Bird Calcium – Hormone × Sex ANOVA
#
# Two-way ANOVA: Ca ~ Hormone * Sex. Both raw and log10 scales.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, anova_table
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, np, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(779, "BirdCalcium", "hormone affect bird calcium", ["Bird", "Sex", "Hormone", "Group", "Ca"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    df = df.with_columns(pl.col("Ca").log10().alias("logCa"))
    means = df.group_by(["Hormone", "Sex"]).agg(pl.col("Ca").mean().round(2)).sort(["Hormone", "Sex"])
    print("Mean Ca by group:")
    print(means)
    df.head(8)
    return df, means

@app.cell
def _(df, fit_lm, anova_table):
    """Two-way ANOVA on raw Ca."""
    model = fit_lm(df, "Ca ~ Hormone * Sex")
    aov = anova_table(model)
    print("ANOVA: Ca ~ Hormone * Sex")
    print(aov)
    return aov, model

@app.cell
def _(df, fit_lm, anova_table):
    """Two-way ANOVA on log10(Ca)."""
    model_log = fit_lm(df, "logCa ~ Hormone * Sex")
    aov_log = anova_table(model_log)
    print("ANOVA: log10(Ca) ~ Hormone * Sex")
    print(aov_log)
    return aov_log, model_log

@app.cell
def _(means, plt):
    """Interaction plot."""
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _sex, _label, _ls in [("male", "Male", "-"), ("female", "Female", "--")]:
        _d = means.filter(pl.col("Sex") == _sex).sort("Hormone")
        _ax1.plot(_d["Hormone"], _d["Ca"], "o-", linestyle=_ls, label=_label)
    _ax1.set_xlabel("Hormone"); _ax1.set_ylabel("Mean Ca"); _ax1.legend()
    for _horm, _label, _ls in [("no", "No hormone", "-"), ("yes", "Hormone", "--")]:
        _d = means.filter(pl.col("Hormone") == _horm).sort("Sex")
        _ax2.plot(_d["Sex"], _d["Ca"], "o-", linestyle=_ls, label=_label)
    _ax2.set_xlabel("Sex"); _ax2.set_ylabel("Mean Ca"); _ax2.legend()
    plt.gca(); return

if __name__ == "__main__": app.run()