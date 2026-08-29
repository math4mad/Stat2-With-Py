# /// notebook
# # ch09 · Med GPA – Logistic Regression
#
# Medical school acceptance: Acceptance ~ GPA (logistic) vs MCAT ~ GPA (linear).
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import polars as pl; import numpy as np; import matplotlib.pyplot as plt; import marimo as mo; import pandas as pd
    from stat2lib.data import load_rda, Stat2Table; from stat2lib.stats import fit_lm, fit_glm
    return Path, Stat2Table, fit_glm, fit_lm, load_rda, mo, np, pd, pl, plt, sys

@app.cell
def _(Stat2Table, mo):
    desc = Stat2Table(1106, "MedGPA", "MedSchool acceptance", ["Accept", "Acceptance", "GPA", "MCAT"])
    mo.md(f"# {desc.question}"); return desc

@app.cell
def _(desc, load_rda):
    data = load_rda(desc.name); data.head(8)
    return data

@app.cell
def _(data, fit_lm, fit_glm):
    model_lm = fit_lm(data, "MCAT ~ GPA")
    model_glm = fit_glm(data, "Acceptance ~ GPA")
    print("=== Logistic: Acceptance ~ GPA ===")
    print(model_glm.summary())
    print(f"\n=== Linear: MCAT ~ GPA ===")
    print(model_lm.summary())
    return model_glm, model_lm

@app.cell
def _(data, model_glm, model_lm, np, pd, plt):
    _xtest = np.linspace(2.5, 4.0, 100)
    _y_lm = model_lm.predict({"GPA": _xtest})
    _y_glm = model_glm.predict(pd.DataFrame({"GPA": _xtest}))
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _ax1.scatter(data["GPA"], data["MCAT"], s=60, c="purple", alpha=0.4, edgecolors="black")
    _ax1.plot(_xtest, _y_lm, "orange", linewidth=2); _ax1.set_xlabel("GPA"); _ax1.set_ylabel("MCAT")
    _ax2.scatter(data["GPA"], data["Acceptance"], s=60, c="purple", alpha=0.4, edgecolors="black")
    _ax2.plot(_xtest, _y_glm, "orange", linewidth=2); _ax2.set_xlabel("GPA"); _ax2.set_ylabel("P(Accept)")
    plt.gca(); return

if __name__ == "__main__": app.run()