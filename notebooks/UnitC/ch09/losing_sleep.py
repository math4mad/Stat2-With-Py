# /// notebook
# # ch09 · Losing Sleep – Logistic Regression
#
# Children sleep: proportion sleeping >7h by age. Linear vs logistic fit.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    import pandas as pd
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, fit_glm
    return Path, Stat2Table, fit_glm, fit_lm, load_rda, mo, np, pd, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1106, "LosingSleep", "children sleep status", ["Age", "Outcome"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda, pl, np):
    df = load_rda(desc.name)
    ages = sorted(df["Age"].unique().to_list())
    rows = []
    for a in ages:
        sub = df.filter(pl.col("Age") == a)
        n = sub.shape[0]; m7 = sub.filter(pl.col("Outcome") == 1).shape[0]
        rows.append({"Age": a, "more7": m7, "less7": n - m7, "total": n, "proportion": round(m7 / n, 2)})
    agg = pl.DataFrame(rows)
    agg
    return agg, df

@app.cell
def _(agg, plt, np):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.scatter(agg["Age"], agg["proportion"], s=80, c="purple", alpha=0.6, edgecolors="black")
    _ax.set_xlabel("Age"); _ax.set_ylabel("P(Sleep > 7h)"); _ax.set_ylim(0, 1)
    plt.gca(); return

@app.cell
def _(agg, df, fit_lm, fit_glm, np, pd, plt):
    model_lm = fit_lm(agg, "proportion ~ Age")
    model_glm = fit_glm(df, "Outcome ~ Age")
    _xtest = np.linspace(0, 40, 100)
    _y_lm = model_lm.predict({"Age": _xtest})
    _y_glm = model_glm.predict(pd.DataFrame({"Age": _xtest})).values
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.scatter(agg["Age"], agg["proportion"], s=80, c="purple", alpha=0.6, edgecolors="black", label="Data")
    _ax.plot(_xtest, _y_lm, "b-", linewidth=2, label="Linear")
    _ax.plot(_xtest, _y_glm, "orange", linewidth=2, label="Logistic")
    _ax.set_xlabel("Age"); _ax.set_ylabel("P(Sleep > 7h)"); _ax.set_ylim(0, 1); _ax.legend()
    plt.gca(); return model_glm, model_lm

if __name__ == "__main__": app.run()