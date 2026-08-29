# /// notebook
# # ch07 · Pig Feed – Two-Way ANOVA with Interaction
#
# Antibiotics + B12 affect pig weight gain? Two-way ANOVA with interaction plot.
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
    desc = Stat2Table(778, "PigFeed", "antibiotics affect pig gain weight by b12", ["WgtGain", "Antibiotic", "B12"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl):
    df = load_rda(desc.name)
    # Contingency table
    means = df.group_by(["Antibiotic", "B12"]).agg(pl.col("WgtGain").mean().round(2)).sort(["Antibiotic", "B12"])
    print("Mean weight gain by group:")
    print(means)
    df.head(8)
    return df, means

@app.cell
def _(df, fit_lm, anova_table):
    """Two-way ANOVA with interaction."""
    model = fit_lm(df, "WgtGain ~ Antibiotic * B12")
    aov = anova_table(model)
    print("Two-way ANOVA: WgtGain ~ Antibiotic * B12")
    print(aov)
    return aov, model

@app.cell
def _(means, plt, np):
    """Interaction plot."""
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _b12_val, _label, _ls in [(0, "B12=0mg", "--"), (5, "B12=5mg", "-")]:
        _d = means.filter(pl.col("B12") == _b12_val).sort("Antibiotic")
        _ax1.plot(_d["Antibiotic"], _d["WgtGain"], "o-", linestyle=_ls, label=_label)
    _ax1.set_xlabel("Antibiotic (mg)"); _ax1.set_ylabel("Mean Weight Gain"); _ax1.legend()
    _ax1.set_xticks([0, 40])
    for _ab_val, _label, _ls in [(0, "Antibiotic=0mg", "--"), (40, "Antibiotic=40mg", "-")]:
        _d = means.filter(pl.col("Antibiotic") == _ab_val).sort("B12")
        _ax2.plot(_d["B12"], _d["WgtGain"], "o-", linestyle=_ls, label=_label)
    _ax2.set_xlabel("B12 (mg)"); _ax2.set_ylabel("Mean Weight Gain"); _ax2.legend()
    _ax2.set_xticks([0, 5])
    plt.gca(); return

if __name__ == "__main__": app.run()