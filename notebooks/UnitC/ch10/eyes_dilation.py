# /// notebook
# # ch10 · Eyes – Logistic Regression
#
# Sexual orientation and pupil dilation: Gay ~ DilateDiff.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; import marimo as mo
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_glm
    return Path, Stat2Table, fit_glm, load_rda, mo, np, pl, plt, sns, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1201, "Eyes", "sexual attitude", ["DilateDiff", "Sex", "Gay", "SexMale"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda):
    df = load_rda(desc.name); df.head(8)
    return df

@app.cell
def _(df, plt, sns):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df.to_pandas(), x="Gay", y="DilateDiff", width=0.4, ax=_ax)
    sns.stripplot(data=df.to_pandas(), x="Gay", y="DilateDiff", color="black", alpha=0.4, size=6, ax=_ax)
    _ax.set_title("Pupil Dilation by Sexual Orientation"); plt.gca(); return

@app.cell
def _(df, fit_glm):
    model = fit_glm(df, "Gay ~ DilateDiff")
    print("Logistic: Gay ~ DilateDiff")
    print(model.summary())
    return model

@app.cell
def _(df, np):
    """Aggregate by dilation range."""
    ranges = [(-1.1, -0.301), (-0.3, -0.074), (-0.073, 0.07), (0.071, 1.3)]
    rows = []
    for r in ranges:
        sub = df.filter((pl.col("DilateDiff") > r[0]) & (pl.col("DilateDiff") < r[1]))
        rows.append({"Range": f"{r[0]}, {r[1]}", "n": sub.shape[0], "mean_diff": sub["DilateDiff"].mean(), "gay_count": sub.filter(pl.col("Gay") == 1).shape[0]})
    agg = pl.DataFrame(rows)
    agg
    return agg

if __name__ == "__main__": app.run()