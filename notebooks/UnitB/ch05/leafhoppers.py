# /// notebook
# # ch05 · Leafhoppers – Diet ANOVA
#
# Does diet affect lifespan of leafhoppers? ANOVA: Days ~ Diet.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, anova_table
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, pl, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(579, "Leafhoppers", "diet affect life long of insect?", ["Diet", "Days"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda):
    df = load_rda(desc.name); df.head(8)
    return df

@app.cell
def _(df, fit_lm, anova_table):
    model = fit_lm(df, "Days ~ Diet")
    aov = anova_table(model)
    print("ANOVA: Days ~ Diet")
    print(aov)
    print(f"\np(Diet) = {model.pvalues.get('Diet[T.Sucrose]', '—'):}")
    return aov, model

if __name__ == "__main__": app.run()