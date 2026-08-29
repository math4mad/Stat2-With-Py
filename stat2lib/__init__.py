"""stat2lib - Python utilities for Stat2 textbook analyses.

Provides data loading, statistical helpers, and publication-quality plotting
for Lock5Stat datasets.
"""

from stat2lib.data import load_rda, Stat2Table, summary_df
from stat2lib.plotting import (
    plot_pair_scatter,
    plot_fitline_and_residual,
    plot_lm_res,
    plot_lm_res2,
    plot_voltage,
)
from stat2lib.stats import fit_lm, anova_table

__all__ = [
    "load_rda",
    "Stat2Table",
    "summary_df",
    "plot_pair_scatter",
    "plot_fitline_and_residual",
    "plot_lm_res",
    "plot_lm_res2",
    "plot_voltage",
    "fit_lm",
    "anova_table",
]