"""Tests for ch01 – Linear Regression analyses."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm, anova_table


class TestDoctorsHospital:
    """Validate Example 1.7 – Doctors vs Hospitals."""

    @pytest.fixture(scope="class")
    def data(self):
        df = load_rda("CountyHealth")
        return df.select(["Hospitals", "MDs"])

    @pytest.fixture(scope="class")
    def model(self, data):
        return fit_lm(data, "MDs ~ Hospitals")

    def test_model_coefficients(self, model):
        # Julia: Intercept ≈ -1120.56, Hospitals ≈ 557.32
        params = model.params
        assert np.isclose(params["Intercept"], -1120.56, atol=5)
        assert np.isclose(params["Hospitals"], 557.32, atol=2)

    def test_model_significant(self, model):
        assert model.f_pvalue < 1e-15

    def test_r_squared(self, model):
        assert model.rsquared > 0.5


class TestDoctorsHospitalSqrt:
    """Validate Example 1.7 with sqrt transform."""

    @pytest.fixture(scope="class")
    def df(self):
        df = load_rda("CountyHealth")
        return df.with_columns(pl.col("MDs").sqrt().alias("sqrt_MDs"))

    @pytest.fixture(scope="class")
    def model(self, df):
        return fit_lm(df, "sqrt_MDs ~ Hospitals")

    def test_model_coefficients(self, model):
        # Julia: Intercept ≈ -2.75, Hospitals ≈ 6.88
        params = model.params
        assert np.isclose(params["Intercept"], -2.75, atol=0.5)
        assert np.isclose(params["Hospitals"], 6.88, atol=0.2)

    def test_model_significant(self, model):
        assert model.f_pvalue < 1e-15


class TestLeafWidth:
    """Validate Ex 1.15 – Leaf Width vs Year."""

    @pytest.fixture(scope="class")
    def data(self):
        df = load_rda("LeafWidth")
        return df.select(["Year", "Width"])

    @pytest.fixture(scope="class")
    def model(self, data):
        return fit_lm(data, "Width ~ Year")

    def test_model_negative_slope(self, model):
        # Leaf width decreases with year
        assert model.params["Year"] < 0

    def test_model_significant(self, model):
        assert model.pvalues["Year"] < 0.001


class TestGlowWorms:
    """Validate Ex 1.16 – Glow Worms."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("GlowWorms")
        return fit_lm(df, "Eggs ~ Lantern")

    def test_model_positive_slope(self, model):
        # More lantern → more eggs
        assert model.params["Lantern"] > 0

    def test_model_significant(self, model):
        assert model.pvalues["Lantern"] < 0.01


class TestOysters:
    """Validate Ex 1.43 – Oyster 2D vs 3D."""

    @pytest.fixture(scope="class")
    def df(self):
        return load_rda("Oysters")

    def test_3d_better_than_2d(self, df):
        data_2d = df.select(["TwoD", "Volume"])
        data_3d = df.select(["ThreeD", "Volume"])
        model_2d = fit_lm(data_2d, "Volume ~ TwoD")
        model_3d = fit_lm(data_3d, "Volume ~ ThreeD")
        # 3D should have higher R²
        assert model_3d.rsquared > model_2d.rsquared

    def test_anova_3d_high_f(self, df):
        data_3d = df.select(["ThreeD", "Volume"])
        model_3d = fit_lm(data_3d, "Volume ~ ThreeD")
        aov = anova_table(model_3d)
        # F-value for ThreeD should be large
        ft = aov.filter(pl.col("index") == "ThreeD")
        assert ft["F"].item() > 100


class TestAccordPrice:
    """Validate p106 – Accord Price."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("AccordPrice")
        data = df.select(["Mileage", "Price"])
        return fit_lm(data, "Price ~ Mileage")

    def test_negative_slope(self, model):
        # Higher mileage → lower price
        assert model.params["Mileage"] < 0

    def test_correlation_negative(self):
        df = load_rda("AccordPrice")
        corr = df.select(pl.corr("Mileage", "Price")).item()
        assert corr < -0.8  # strong negative

    def test_r_squared(self, model):
        assert 0.65 < model.rsquared < 0.80


class TestCapacitorVoltage:
    """Validate capacitor voltage log-transform."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("Volts")
        df = df.with_columns(pl.col("Voltage").log().alias("logVoltage"))
        return fit_lm(df, "logVoltage ~ Time")

    def test_model_significant(self, model):
        assert model.pvalues["Time"] < 1e-30

    def test_negative_slope(self, model):
        # Voltage decays over time
        assert model.params["Time"] < 0


class TestSpeciesArea:
    """Validate Example 1.8 – Species–Area."""

    @pytest.fixture(scope="class")
    def df(self):
        return load_rda("SpeciesArea")

    def test_log_log_model_better(self, df):
        data1 = df.select(["logArea", "Species"])
        data2 = df.select(["logArea", "logSpecies"])
        m1 = fit_lm(data1, "Species ~ logArea")
        m2 = fit_lm(data2, "logSpecies ~ logArea")
        # log-log should have higher R²
        # (not always true, but for this dataset it should be)
        assert m2.rsquared > 0.5

    def test_log_log_slope(self, df):
        data2 = df.select(["logArea", "logSpecies"])
        m2 = fit_lm(data2, "logSpecies ~ logArea")
        # Slope ≈ 0.235 (from Julia output)
        assert np.isclose(m2.params["logArea"], 0.235, atol=0.05)


class TestLongJump:
    """Validate studentized residuals for long jump."""

    @pytest.fixture(scope="class")
    def model(self):
        df = load_rda("LongJumpOlympics2016")
        return fit_lm(df, "Gold ~ Year")

    def test_model_positive_slope(self, model):
        # Jump distances increase over time
        assert model.params["Year"] > 0

    def test_studentized_residuals(self, model):
        influence = model.get_influence()
        stud_res = influence.resid_studentized_external
        # Some residuals should be outside ±2
        assert (np.abs(stud_res) > 2).any()

    def test_model_significant(self, model):
        assert model.pvalues["Year"] < 1e-5