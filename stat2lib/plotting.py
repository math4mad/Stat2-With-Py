"""Publication-quality plotting utilities for Stat2 analyses.

Uses matplotlib + seaborn for static figures suitable for print.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import seaborn as sns
from statsmodels.regression.linear_model import RegressionResultsWrapper

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _to_numpy(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract a column as a numpy array."""
    return df[col].to_numpy()


def _add_border(ax: plt.Axes, color: str = "orange", alpha: float = 0.1) -> None:
    """Add a subtle background box to an Axes."""
    ax.patch.set_facecolor(color)
    ax.patch.set_alpha(alpha)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_pair_scatter(
    data: pl.DataFrame,
    *,
    xlabel: str,
    ylabel: str,
    save: bool = False,
    savepath: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter plot of two DataFrame columns.

    Parameters
    ----------
    data : pl.DataFrame
    xlabel : str
        Predictor column name.
    ylabel : str
        Response column name.
    save : bool
        If True, save to file.
    savepath : str, optional
        Custom save path.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    _add_border(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{xlabel}–{ylabel} scatter")

    ax.scatter(
        _to_numpy(data, xlabel),
        _to_numpy(data, ylabel),
        s=80,
        c="purple",
        alpha=0.4,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    if save:
        path = savepath or f"{xlabel}-{ylabel}-scatter.png"
        fig.savefig(path, bbox_inches="tight")

    return fig, ax


def plot_fitline_and_residual(
    *,
    data: pl.DataFrame,
    xlabel: str,
    ylabel: str,
    model: RegressionResultsWrapper,
    save: bool = False,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """Two-panel plot: (left) scatter + fitted line, (right) residuals stem.

    Parameters
    ----------
    data : pl.DataFrame
    xlabel : str
    ylabel : str
    model : statsmodels RegressionResultsWrapper
    save : bool
    savepath : str, optional

    Returns
    -------
    fig
    """
    x = _to_numpy(data, xlabel)
    y = _to_numpy(data, ylabel)
    y_hat = model.predict()
    res = model.resid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax in (ax1, ax2):
        _add_border(ax)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("Residuals")

    ax1.scatter(
        x, y, s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8
    )
    ax1.plot(x, y_hat, color="blue", linewidth=2, label="Fit line")
    ax1.legend()

    ax2.stem(res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax2.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()

    if save:
        path = savepath or f"{xlabel}-{ylabel}-fit-residual.png"
        fig.savefig(path, bbox_inches="tight")

    return fig


def plot_lm_res(
    *,
    data: pl.DataFrame,
    xlabel: str,
    ylabel: str,
    model: RegressionResultsWrapper,
    save: bool = False,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """Four-panel regression diagnostic plot.

    (1) scatter + fitted line
    (2) residuals vs fitted
    (3) histogram of residuals
    (4) Q-Q plot of residuals

    Parameters
    ----------
    data : pl.DataFrame
    xlabel : str
    ylabel : str
    model : statsmodels RegressionResultsWrapper
    save : bool
    savepath : str, optional

    Returns
    -------
    fig
    """
    x = _to_numpy(data, xlabel)
    y = _to_numpy(data, ylabel)
    y_hat = model.predict()
    res = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax1, ax2), (ax3, ax4) = axes
    fig.suptitle(f"{xlabel}–{ylabel} Linear Regression Diagnostics", fontsize=14)

    for ax in (ax1, ax2, ax3, ax4):
        _add_border(ax)

    # (1) Scatter + fit
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.scatter(
        x, y, s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8
    )
    ax1.plot(x, y_hat, color="blue", linewidth=2, label="Fit line")
    ax1.legend()

    # (2) Residuals vs fitted
    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("Residuals")
    ax2.stem(y_hat, res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax2.axhline(0, color="black", linewidth=0.8)

    # (3) Histogram
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.hist(res, bins=15, color="gray", edgecolor="black", linewidth=0.8)

    # (4) Q-Q plot
    ax4.set_xlabel("Theoretical quantiles")
    ax4.set_ylabel("Sample quantiles")
    stats.probplot(res, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor("purple")
    ax4.get_lines()[0].set_markeredgecolor("black")
    ax4.get_lines()[0].set_alpha(0.6)

    fig.tight_layout()

    if save:
        path = savepath or f"{xlabel}-{ylabel}-lm-res.png"
        fig.savefig(path, bbox_inches="tight")

    return fig


def plot_lm_res2(
    *,
    data: pl.DataFrame,
    xlabel: str,
    ylabel: str,
    model: RegressionResultsWrapper,
    save: bool = False,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """Four-panel diagnostic plot with squared response fitted line.

    Same layout as ``plot_lm_res``, but the fitted line in panel (1)
    uses squared fitted values.

    Parameters
    ----------
    data : pl.DataFrame
    xlabel : str
    ylabel : str
    model : statsmodels RegressionResultsWrapper
    save : bool
    savepath : str, optional

    Returns
    -------
    fig
    """
    x = _to_numpy(data, xlabel)
    y = _to_numpy(data, ylabel)
    y_hat = model.predict() ** 2
    res = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax1, ax2), (ax3, ax4) = axes
    fig.suptitle(f"{xlabel}–{ylabel} Linear Regression (sq) Diagnostics", fontsize=14)

    for ax in (ax1, ax2, ax3, ax4):
        _add_border(ax)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    # Sort by x for a clean line
    sort_idx = np.argsort(x)
    ax1.scatter(
        x, y**2, s=60, c="purple", alpha=0.4, edgecolors="black", linewidths=0.8
    )
    ax1.plot(x[sort_idx], y_hat[sort_idx], color="blue", linewidth=2, label="Fit line")
    ax1.legend()

    ax2.set_xlabel("Fitted values")
    ax2.set_ylabel("Residuals")
    ax2.stem(y_hat, res, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax2.axhline(0, color="black", linewidth=0.8)

    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Frequency")
    ax3.hist(res, bins=15, color="gray", edgecolor="black", linewidth=0.8)

    ax4.set_xlabel("Theoretical quantiles")
    ax4.set_ylabel("Sample quantiles")
    stats.probplot(res, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor("purple")
    ax4.get_lines()[0].set_markeredgecolor("black")
    ax4.get_lines()[0].set_alpha(0.6)

    fig.tight_layout()

    if save:
        path = savepath or f"{xlabel}-{ylabel}-lm-res-sq.png"
        fig.savefig(path, bbox_inches="tight")

    return fig


def plot_voltage(
    df: pl.DataFrame,
    save: bool = False,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side scatter of Time vs Voltage and Time vs log(Voltage).

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``Time``, ``Voltage``, ``logVoltage`` columns.
    save : bool
    savepath : str, optional

    Returns
    -------
    fig
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax in (ax1, ax2):
        _add_border(ax)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Voltage")
    ax1.scatter(
        _to_numpy(df, "Time"),
        _to_numpy(df, "Voltage"),
        s=60, c="red", alpha=0.4, edgecolors="purple", linewidths=1.2,
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("log(Voltage)")
    ax2.scatter(
        _to_numpy(df, "Time"),
        _to_numpy(df, "logVoltage"),
        s=60, c="red", alpha=0.4, edgecolors="purple", linewidths=1.2,
    )

    fig.tight_layout()

    if save:
        path = savepath or "time-voltage-logvoltage-scatter.png"
        fig.savefig(path, bbox_inches="tight")

    return fig