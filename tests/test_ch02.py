"""Tests for ch02 – Linear Regression (Inference)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm, anova_table


class TestTextPrices:
    """Book pages vs price."""

    @pytest.fixture(scope="class")
    def model(self):
        data = load_rda("TextPrices")
        return fit_lm(data, "Price ~ Pages")

    def test_positive_slope(self, model):
        assert model.params["Pages"] > 0

    def test_significant(self, model):
        assert model.pvalues["Pages"] < 1e-5


class TestCereal:
    """Cereal sugar vs calories."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("Cereal")
        data = df.select(["Calories", "Sugar"])
        return fit_lm(data, "Calories ~ Sugar")

    def test_positive_slope(self, model):
        # More sugar → more calories
        assert model.params["Sugar"] > 0

    def test_significant(self, model):
        assert model.pvalues["Sugar"] < 0.01


class TestButterfliesBc:
    """Butterfly wing size vs temperature."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("ButterfliesBc")
        first_sex = df["Sex"].unique().sort().item(0)
        data = df.filter(pl.col("Sex") == first_sex).select(["Temp", "Wing"])
        return fit_lm(data, "Wing ~ Temp")

    def test_negative_slope(self, model):
        # Warmer → smaller wings
        assert model.params["Temp"] < 0

    def test_significant(self, model):
        assert model.pvalues["Temp"] < 0.05

    def test_r_squared(self, model):
        # Julia: R² ≈ 0.392
        assert 0.35 < model.rsquared < 0.45


class TestMetabolicRate:
    """Metabolic rate log-log regression."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("MetabolicRate")
        data = df.select(["LogBodySize", "LogMrate"])
        return fit_lm(data, "LogMrate ~ LogBodySize")

    def test_positive_slope(self, model):
        assert model.params["LogBodySize"] > 0

    def test_high_r_squared(self, model):
        # Julia: R² ≈ 0.948
        assert model.rsquared > 0.9

    def test_anova(self, model):
        aov = anova_table(model)
        assert aov.shape[0] >= 2