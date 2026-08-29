# /// notebook
# # ch03 · NFL – Offense & Defense vs Win %
#
# Multiple regression: WinPct ~ PointsFor + PointsAgainst.
# Includes ANOVA table and R².
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import polars as pl
    import matplotlib.pyplot as plt
    import marimo as mo
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.stats import fit_lm, anova_table
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, pl, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=293,
        name="NFLStandings2016",
        question="defense offense which is important",
        feature=["PointsFor", "PointsAgainst", "WinPct"],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda):
    """1. Load data."""
    data = load_rda(desc.name)
    data.head(8)
    return data


@app.cell
def _(data, fit_lm):
    """2. Fit model: WinPct ~ PointsFor + PointsAgainst."""
    model = fit_lm(data, "WinPct ~ PointsFor + PointsAgainst")
    print(model.summary())
    print(f"\nR² = {model.rsquared:.4f}")
    return model


@app.cell
def _(model, anova_table):
    """3. ANOVA table."""
    aov = anova_table(model)
    print("ANOVA Table:")
    print(aov)
    return aov


if __name__ == "__main__":
    app.run()