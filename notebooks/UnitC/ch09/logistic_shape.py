# /// notebook
# # ch09 · Logistic Shape – Parameter Effects
#
# How β₀ and β₁ affect the shape of the logistic curve.
# ///

import marimo; __generated_with = "0.24.0"; app = marimo.App()

@app.cell
def _():
    import numpy as np; import matplotlib.pyplot as plt; import marimo as mo
    from stat2lib.stats import logistic
    return logistic, mo, np, plt

@app.cell
def _(mo):
    mo.md("# Logistic Curve Shape – Parameter Effects"); return

@app.cell
def _(logistic, np, plt):
    _xs1 = np.linspace(0, 4.5, 100); _xs2 = np.linspace(0, 20, 100)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for _b0, _ls in [(-17, ":"), (-19, "-"), (-21, "-.")]:
        _ax1.plot(_xs1, logistic(5.5 * _xs1 + _b0), linestyle=_ls, linewidth=3, label=f"β₀={_b0}")
    _ax1.set_title("Change β₀, with β₁=5.5"); _ax1.legend()
    for _b1, _ls in [(-0.8, ":"), (-0.5, "-"), (-0.3, "-.")]:
        _ax2.plot(_xs2, logistic(_b1 * _xs2 + 4), linestyle=_ls, linewidth=3, label=f"β₁={_b1}")
    _ax2.set_title("Change β₁, with β₀=4"); _ax2.legend()
    plt.gca(); return

if __name__ == "__main__": app.run()