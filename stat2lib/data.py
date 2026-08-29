"""Data loading utilities for Stat2 datasets.

Provides functions to load RDA datasets into polars DataFrames and
a Stat2Table descriptor for metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import polars as pl
import pyreadr


# ---------------------------------------------------------------------------
# Resolve the project root and Stat2Data directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_stat2data_dir() -> Path:
    """Locate the Stat2Data directory relative to the project root."""
    candidates = [
        _PROJECT_ROOT / "Stat2Data",
        _PROJECT_ROOT / ".." / "Stat2Data",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError("Cannot locate Stat2Data directory")


_STAT2DATA = _find_stat2data_dir()


@dataclass
class Stat2Table:
    """Descriptor for a Stat2 textbook dataset / table.

    Attributes
    ----------
    page : int
        Textbook page number.
    name : str
        Dataset name (without .rda extension).
    question : str
        Description of the research question.
    feature : list of str
        Column names of interest.
    """

    page: int
    name: str
    question: str
    feature: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_rda(name: str) -> pl.DataFrame:
    """Load a Stat2 RDA dataset and return it as a polars DataFrame.

    Parameters
    ----------
    name : str
        Dataset name without the ``.rda`` extension
        (e.g. ``"WeightLossIncentive4"``).

    Returns
    -------
    pl.DataFrame
    """
    rda_path = _STAT2DATA / f"{name}.rda"
    if not rda_path.exists():
        raise FileNotFoundError(f"RDA file not found: {rda_path}")
    r_obj = pyreadr.read_r(str(rda_path))
    if name not in r_obj:
        # Fallback: try the first key
        key = list(r_obj.keys())[0]
    else:
        key = name
    return pl.from_pandas(r_obj[key])


def summary_df(
    data: pl.DataFrame, group_col: str, value_col: str
) -> pl.DataFrame:
    """Compute grouped summary statistics (n, mean, std).

    Parameters
    ----------
    data : pl.DataFrame
    group_col : str
        Column to group by.
    value_col : str
        Column to summarise.

    Returns
    -------
    pl.DataFrame
        Columns: ``group_col``, ``n``, ``Mean``, ``Stddev``.
    """
    return (
        data.group_by(group_col)
        .agg(
            pl.col(value_col).len().alias("n"),
            pl.col(value_col).mean().alias("Mean"),
            pl.col(value_col).std().alias("Stddev"),
        )
        .sort(group_col)
    )


def list_features(df: pl.DataFrame) -> None:
    """Print column names of a DataFrame."""
    print(df.columns)


def peek(df: pl.DataFrame, n: int = 5) -> pl.DataFrame:
    """Return the first *n* rows of a DataFrame."""
    return df.head(n)