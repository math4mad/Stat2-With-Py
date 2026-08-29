"""Tests for stat2lib.data – data loading utilities."""

from __future__ import annotations

import polars as pl
import pytest
from stat2lib.data import load_rda, Stat2Table, summary_df, peek, list_features


class TestLoadRda:
    """Test RDA data loading."""

    def test_load_weightloss(self):
        df = load_rda("WeightLossIncentive4")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] > 0
        assert "Group" in df.columns
        assert "WeightLoss" in df.columns

    def test_load_countyhealth(self):
        df = load_rda("CountyHealth")
        assert isinstance(df, pl.DataFrame)
        assert "MDs" in df.columns
        assert "Hospitals" in df.columns

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_rda("NonExistentDataset")

    def test_load_accordprice(self):
        df = load_rda("AccordPrice")
        assert "Price" in df.columns
        assert "Mileage" in df.columns
        assert "Age" in df.columns


class TestStat2Table:
    """Test Stat2Table descriptor."""

    def test_create(self):
        desc = Stat2Table(
            page=73,
            name="WeightLossIncentive4",
            question="Financial incentives for weight loss",
            feature=["Group", "WeightLoss"],
        )
        assert desc.page == 73
        assert desc.name == "WeightLossIncentive4"
        assert len(desc.feature) == 2

    def test_default_feature(self):
        desc = Stat2Table(page=138, name="CountyHealth", question="Test")
        assert desc.feature == []


class TestSummaryDf:
    """Test summary statistics."""

    def test_summary_weightloss(self):
        df = load_rda("WeightLossIncentive4")
        summary = summary_df(df, "Group", "WeightLoss")
        assert summary.shape[0] == 2  # Control, Incentive
        assert "n" in summary.columns
        assert "Mean" in summary.columns
        assert "Stddev" in summary.columns

    def test_peek(self):
        df = load_rda("WeightLossIncentive4")
        result = peek(df, n=3)
        assert result.shape[0] == 3

    def test_list_features(self, capsys):
        df = load_rda("AccordPrice")
        list_features(df)
        captured = capsys.readouterr()
        assert "Price" in captured.out