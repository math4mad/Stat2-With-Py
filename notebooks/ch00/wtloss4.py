# /// notebook
# # ch00 · Weight Loss Incentive Study
#
# This notebook replicates the Julia analysis in `ch00/wtloss4.jl`.
# It compares weight loss between a control group and a group that
# received financial incentives.
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    """Imports and path setup."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from stat2lib.data import load_rda, Stat2Table, summary_df
    from stat2lib.plotting import _add_border as add_border
    from stat2lib.stats import t_test_ind, summary_table
    import marimo as mo
    return (
        Path, Stat2Table, add_border, load_rda, mo, np, pl, plt, sns,
        summary_df, summary_table, sys, t_test_ind,
    )


@app.cell
def _(Stat2Table, mo):
    """Title."""
    desc = Stat2Table(
        page=73,
        name="WeightLossIncentive4",
        question="Financial incentives for weight loss",
        feature=[],
    )
    mo.md(f"# {desc.question}")
    return desc


@app.cell
def _(desc, load_rda, pl):
    """1. Load data."""
    df = load_rda(desc.name)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    df.head(8)
    return df


@app.cell
def _(df, np, pl, summary_table):
    """2. Group summary statistics."""
    control = df.filter(pl.col("Group") == "Control")
    incentive = df.filter(pl.col("Group") == "Incentive")
    control_wl = control["WeightLoss"].to_numpy()
    incentive_wl = incentive["WeightLoss"].to_numpy()
    print(
        f"Control:   n={len(control_wl)}, "
        f"mean={control_wl.mean():.4f}, "
        f"std={control_wl.std(ddof=1):.4f}"
    )
    print(
        f"Incentive: n={len(incentive_wl)}, "
        f"mean={incentive_wl.mean():.4f}, "
        f"std={incentive_wl.std(ddof=1):.4f}"
    )
    summary_table(df, "Group", "WeightLoss")
    return control_wl, incentive_wl


@app.cell
def _(control_wl, df, incentive_wl, plt, sns, add_border):
    """3. Side-by-side boxplot and histogram."""
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 5))
    sns.boxplot(
        data=df.to_pandas(), x="Group", y="WeightLoss",
        palette={"Control": "#7f7f7f", "Incentive": "#ff7f0e"},
        width=0.5, ax=_ax1,
    )
    sns.stripplot(
        data=df.to_pandas(), x="Group", y="WeightLoss",
        color="black", alpha=0.5, size=6, ax=_ax1,
    )
    _ax1.set_title("Weight Loss by Group")
    add_border(_ax1)
    for _label, _arr, _color in [
        ("Control", control_wl, "#7f7f7f"),
        ("Incentive", incentive_wl, "#ff7f0e"),
    ]:
        _ax2.hist(_arr, bins=8, alpha=0.5, label=_label, color=_color, edgecolor="black")
    _ax2.set_xlabel("Weight Loss")
    _ax2.set_ylabel("Frequency")
    _ax2.set_title("Distribution of Weight Loss")
    _ax2.legend()
    add_border(_ax2)
    _fig.tight_layout()
    plt.gca()
    return


@app.cell
def _(control_wl, incentive_wl, np, t_test_ind):
    """4. Two-sample t-test (equal variance)."""
    result = t_test_ind(control_wl, incentive_wl, equal_var=True)
    print("Two sample t-test (equal variance)")
    print("─" * 50)
    print("  parameter of interest:   Mean difference")
    print("  value under H₀:          0")
    print(f"  point estimate:          {result['mean_diff']:.4f}")
    print(
        f"  95% confidence interval: "
        f"({result['ci_lower']:.4f}, {result['ci_upper']:.4f})"
    )
    print()
    print("Test summary:")
    print(
        f"  outcome with 95% confidence: "
        f"{'reject H₀' if result['pvalue'] < 0.05 else 'fail to reject H₀'}"
    )
    print(f"  two-sided p-value:           {result['pvalue']:.4f}")
    print()
    print("Details:")
    print(f"  number of observations:   [{len(control_wl)}, {len(incentive_wl)}]")
    print(f"  t-statistic:              {result['statistic']:.6f}")
    print(f"  degrees of freedom:       {result['df']}")
    print(
        f"  empirical standard error: "
        f"{np.abs(result['mean_diff'] / result['statistic']):.4f}"
    )
    return result


@app.cell
def _():
    """5. Conclusion."""
    # The incentive group lost significantly more weight (mean ≈ 15.68 lbs)
    # than the control group (mean ≈ 3.92 lbs), p = 0.0006.
    # This suggests financial incentives are effective for weight loss.
    return


if __name__ == "__main__":
    app.run()