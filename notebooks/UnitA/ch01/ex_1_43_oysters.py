# /// notebook
# # ch01 · Ex 1.43 – Oyster 2D vs 3D Measurement
#
# Compare 2D and 3D measurement systems for estimating oyster volume.
# Which method provides a more accurate prediction?
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
    from stat2lib.data import load_rda, Stat2Table
    from stat2lib.plotting import plot_lm_res
    from stat2lib.stats import fit_lm, anova_table
    import marimo as mo
    return Path, Stat2Table, anova_table, fit_lm, load_rda, mo, pl, plot_lm_res, plt, sys


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=191,
        name="Oysters",
        question="2D vs 3D measurement accuracy for oyster volume",
        feature=[],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    data_2d = df.select(["TwoD", "Volume"])
    data_3d = df.select(["ThreeD", "Volume"])
    print(f"Shape: {df.shape}")
    df.head(5)
    return data_2d, data_3d, df


@app.cell
def _(data_2d, fit_lm):
    """2. 2D model."""
    model_2d = fit_lm(data_2d, "Volume ~ TwoD")
    print("=== 2D Model ===")
    print(model_2d.summary())
    return model_2d


@app.cell
def _(data_3d, fit_lm):
    """3. 3D model."""
    model_3d = fit_lm(data_3d, "Volume ~ ThreeD")
    print("=== 3D Model ===")
    print(model_3d.summary())
    return model_3d


@app.cell
def _(anova_table, model_2d, model_3d):
    """4. ANOVA tables."""
    print("=== ANOVA: 2D ===")
    print(anova_table(model_2d))
    print("\n=== ANOVA: 3D ===")
    print(anova_table(model_3d))
    return


@app.cell
def _(data_2d, data_3d, model_2d, model_3d, plot_lm_res, plt):
    """5. Diagnostic plots."""
    _fig1 = plot_lm_res(data=data_2d, xlabel="TwoD", ylabel="Volume", model=model_2d)
    _fig1.suptitle("2D Measurement Diagnostics")
    plt.gca()
    _fig2 = plot_lm_res(data=data_3d, xlabel="ThreeD", ylabel="Volume", model=model_3d)
    _fig2.suptitle("3D Measurement Diagnostics")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()