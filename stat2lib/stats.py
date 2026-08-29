"""Statistical helpers for Stat2 analyses.

Wraps statsmodels OLS and scipy hypothesis tests in a simple API
familiar to users coming from Julia's GLM / HypothesisTests ecosystem.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import polars as pl
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import great_tables as gt
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing import List


def fit_lm(
    data: pl.DataFrame,
    formula: str,
) -> RegressionResultsWrapper:
    """Fit an ordinary least-squares regression model.

    Parameters
    ----------
    data : pl.DataFrame
    formula : str
        Patsy-style formula, e.g. ``"WeightLoss ~ Group"``
        or ``"MDs ~ Hospitals"``.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    # Convert to pandas for formula API
    pdf = data.to_pandas()
    model = smf.ols(formula=formula, data=pdf)
    return model.fit()


def anova_table(fit: RegressionResultsWrapper) -> pl.DataFrame:
    """Return an ANOVA (Type I) table for a fitted OLS model.

    Parameters
    ----------
    fit : RegressionResultsWrapper

    Returns
    -------
    pl.DataFrame
    """
    # statsmodels does not have a native anova_lm; construct one
    from statsmodels.stats.anova import anova_lm

    anova = anova_lm(fit, typ=1)
    df = pl.from_pandas(anova.reset_index())
    return df


def t_test_ind(
    *groups: np.ndarray,
    equal_var: bool = True,
) -> dict:
    """Two-sample t-test (independent).

    Parameters
    ----------
    *groups : np.ndarray
        Two (or more) arrays. The first two are compared.
    equal_var : bool
        Assume equal variance (Student's t-test).

    Returns
    -------
    dict
        Keys: ``statistic``, ``pvalue``, ``df``, ``mean_diff``,
        ``ci_lower``, ``ci_upper``.
    """
    if len(groups) < 2:
        raise ValueError("Need at least two groups")
    a, b = groups[0], groups[1]
    result = stats.ttest_ind(a, b, equal_var=equal_var)

    # Confidence interval
    n1, n2 = len(a), len(b)
    mean_diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1) / n1 + np.var(b, ddof=1) / n2)
    df = result.df
    t_crit = stats.t.ppf(0.975, df)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df": result.df,
        "mean_diff": mean_diff,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def summary_table(
    data: pl.DataFrame,
    group_col: str,
    value_col: str,
) -> gt.GT:
    """Create a publication-ready summary table with great_tables.

    Parameters
    ----------
    data : pl.DataFrame
    group_col : str
    value_col : str

    Returns
    -------
    great_tables.GT
    """
    summary = (
        data.group_by(group_col)
        .agg(
            pl.col(value_col).len().alias("n"),
            pl.col(value_col).mean().alias("Mean"),
            pl.col(value_col).std().alias("Stddev"),
        )
        .sort(group_col)
    )
    return (
        gt.GT(summary.to_pandas())
        .tab_header(title=f"Summary of {value_col} by {group_col}")
        .fmt_number(columns=["Mean", "Stddev"], decimals=2)
    )


def correlation_matrix(
    data: pl.DataFrame,
    columns: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Compute Pearson correlation matrix for selected columns.

    Parameters
    ----------
    data : pl.DataFrame
    columns : list of str, optional
        Defaults to all numeric columns.

    Returns
    -------
    pl.DataFrame
    """
    if columns is None:
        columns = [c for c in data.columns if data[c].dtype.is_numeric()]
    corr = data.select(columns).to_pandas().corr()
    return pl.from_pandas(corr.reset_index().rename(columns={"index": "variable"}))


# ---------------------------------------------------------------------------
# ANOVA helpers
# ---------------------------------------------------------------------------


def one_way_anova(*groups: np.ndarray) -> dict:
    """One-way ANOVA test.

    Parameters
    ----------
    *groups : np.ndarray
        Two or more arrays representing group values.

    Returns
    -------
    dict
        Keys: ``statistic`` (F), ``pvalue``, ``df_between``, ``df_within``.
    """
    result = stats.f_oneway(*groups)
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    return {
        "statistic": result.statistic,
        "pvalue": result.pvalue,
        "df_between": k - 1,
        "df_within": n_total - k,
    }


def levene_test(*groups: np.ndarray, center: str = "median") -> dict:
    """Levene's test for equality of variances.

    Parameters
    ----------
    *groups : np.ndarray
    center : str
        "median" (default, robust) or "mean".

    Returns
    -------
    dict
        Keys: ``statistic`` (W), ``pvalue``.
    """
    result = stats.levene(*groups, center=center)
    return {"statistic": result.statistic, "pvalue": result.pvalue}


def two_way_anova(
    data: pl.DataFrame,
    formula: str,
) -> RegressionResultsWrapper:
    """Fit a two-way ANOVA model (with interaction if specified).

    Parameters
    ----------
    data : pl.DataFrame
    formula : str
        e.g. ``"WgtGain ~ Antibiotic * B12"`` or ``"Rate ~ Drug + Subj"``.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    return fit_lm(data, formula)


def contingency_table(
    data: pl.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    agg: str = "mean",
) -> pl.DataFrame:
    """Build a two-way contingency / means table.

    Parameters
    ----------
    data : pl.DataFrame
    row_col : str
    col_col : str
    value_col : str
    agg : str
        "mean" or "sum".

    Returns
    -------
    pl.DataFrame
        Pivoted table with row_col as rows, col_col as columns.
    """
    if agg == "mean":
        grouped = data.group_by([row_col, col_col]).agg(pl.col(value_col).mean())
    else:
        grouped = data.group_by([row_col, col_col]).agg(pl.col(value_col).sum())
    return grouped.sort([row_col, col_col])


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


def fit_glm(
    data: pl.DataFrame,
    formula: str,
    family: str = "binomial",
) -> object:
    """Fit a Generalized Linear Model (logistic regression).

    Parameters
    ----------
    data : pl.DataFrame
    formula : str
        e.g. ``"Acceptance ~ GPA"``.
    family : str
        "binomial" (default).

    Returns
    -------
    statsmodels GLM Results object
    """
    pdf = data.to_pandas()
    import patsy
    y, X = patsy.dmatrices(formula, data=pdf, return_type="dataframe")
    if family == "binomial":
        model = sm.GLM(y, X, family=sm.families.Binomial())
    else:
        raise ValueError(f"Unsupported family: {family}")
    return model.fit()


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic (sigmoid) function: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))