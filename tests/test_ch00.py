"""Tests for ch00 – Weight Loss Incentive analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda, summary_df
from stat2lib.stats import t_test_ind


class TestWeightLossIncentive:
    """Validate the ch00 weight loss analysis."""

    @pytest.fixture(scope="class")
    def df(self):
        return load_rda("WeightLossIncentive4")

    @pytest.fixture(scope="class")
    def groups(self, df):
        control = df.filter(pl.col("Group") == "Control")["WeightLoss"].to_numpy()
        incentive = df.filter(pl.col("Group") == "Incentive")["WeightLoss"].to_numpy()
        return control, incentive

    def test_data_shape(self, df):
        assert df.shape[0] == 36  # 19 + 17
        assert "Group" in df.columns
        assert "WeightLoss" in df.columns

    def test_group_counts(self, groups):
        control, incentive = groups
        assert len(control) == 19
        assert len(incentive) == 17

    def test_group_means(self, groups):
        control, incentive = groups
        # From Julia output: Control ≈ 3.92, Incentive ≈ 15.68
        assert np.isclose(np.mean(control), 3.921, atol=0.01)
        assert np.isclose(np.mean(incentive), 15.676, atol=0.01)

    def test_summary_table(self, df):
        summary = summary_df(df, "Group", "WeightLoss")
        row_control = summary.filter(pl.col("Group") == "Control")
        assert row_control["n"].item() == 19

    def test_t_test_equal_var(self, groups):
        control, incentive = groups
        result = t_test_ind(control, incentive, equal_var=True)
        # From Julia: p ≈ 0.0006, t ≈ -3.805
        assert result["pvalue"] < 0.001
        assert np.isclose(abs(result["statistic"]), 3.805, atol=0.01)

    def test_t_test_unequal_var(self, groups):
        control, incentive = groups
        result = t_test_ind(control, incentive, equal_var=False)
        # Welch's t-test should still be significant
        assert result["pvalue"] < 0.01

    def test_ci_contains_mean_diff(self, groups):
        control, incentive = groups
        result = t_test_ind(control, incentive, equal_var=True)
        assert result["ci_lower"] < result["mean_diff"] < result["ci_upper"]