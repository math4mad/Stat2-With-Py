"""Tests for UnitD – Time Series (ch12)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm


class TestSeaIce:
    def test_linear_trend(self):
        df = load_rda("SeaIce")
        model = fit_lm(df, "Extent ~ Year")
        assert model.params["Year"] < 0  # declining trend

    def test_r_squared(self):
        df = load_rda("SeaIce")
        model = fit_lm(df, "Extent ~ Year")
        assert model.rsquared > 0.7


class TestPeaceBridge2012:
    def test_seasonal_model(self):
        df = load_rda("PeaceBridge2012")
        cost = np.cos(2 * np.pi * df["t"].to_numpy() / 12)
        sint = np.sin(2 * np.pi * df["t"].to_numpy() / 12)
        df = df.with_columns([pl.Series("cost", cost), pl.Series("sint", sint)])
        model = fit_lm(df, "Traffic ~ cost + sint")
        assert model.rsquared > 0.5


class TestPeaceBridge2003:
    def test_long_seasonal(self):
        df = load_rda("PeaceBridge2003")
        cost = np.cos(2 * np.pi * df["t"].to_numpy() / 12)
        sint = np.sin(2 * np.pi * df["t"].to_numpy() / 12)
        df = df.with_columns([pl.Series("cost", cost), pl.Series("sint", sint)])
        model = fit_lm(df, "Traffic ~ t + cost + sint")
        assert model.rsquared > 0.7


class TestAppleStock:
    def test_data(self):
        df = load_rda("AppleStock")
        assert "Price" in df.columns
        assert df.shape[0] > 50

    def test_acf_positive_lag1(self):
        df = load_rda("AppleStock")
        price = df["Price"].to_numpy()
        diff = price[1:] - price[:-1]
        acf1 = np.corrcoef(diff[:-1], diff[1:])[0, 1]
        assert abs(acf1) < 0.5  # differencing should reduce autocorrelation