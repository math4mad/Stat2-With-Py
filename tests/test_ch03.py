"""Tests for ch03 – Multiple Regression."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm, anova_table


class TestNFL:
    @pytest.fixture(scope="class")
    def model(self):
        data = load_rda("NFLStandings2016")
        return fit_lm(data, "WinPct ~ PointsFor + PointsAgainst")

    def test_r_squared(self, model):
        assert model.rsquared > 0.7  # Julia: ~0.782

    def test_anova(self, model):
        aov = anova_table(model)
        assert aov.shape[0] >= 3


class TestFish:
    def test_cross_feature_better(self):
        df = load_rda("Perch")
        data = df.with_columns((pl.col("Length") * pl.col("Width")).alias("lengthxwidth"))
        m1 = fit_lm(data, "Weight ~ lengthxwidth")
        m2 = fit_lm(data, "Weight ~ Length + Width + lengthxwidth")
        assert m2.rsquared > m1.rsquared
        assert m1.rsquared > 0.9  # ~0.978


class TestCO2:
    def test_quadratic(self):
        df = load_rda("CO2Germany")
        data = df.with_columns(pl.col("Day").pow(2).alias("Day2"))
        model = fit_lm(data, "CO2 ~ Day + Day2")
        assert model.rsquared > 0.5


class TestFunnelDrop:
    def test_quadratic_model(self):
        df = load_rda("FunnelDrop")
        data = df.with_columns([
            pl.col("Funnel").pow(2).alias("Funnel2"),
            pl.col("Tube").pow(2).alias("Tube2"),
            (pl.col("Funnel") * pl.col("Tube")).alias("Funnel_Tube"),
        ])
        model = fit_lm(data, "Time ~ Funnel + Tube + Funnel2 + Tube2 + Funnel_Tube")
        assert model.rsquared > 0.2


class TestHousesNY:
    def test_multicollinearity(self):
        df = load_rda("HousesNY")
        data = df.select(["Beds", "Baths", "Size", "Lot", "Price"])
        model = fit_lm(data, "Price ~ Beds + Baths + Size + Lot")
        assert model.rsquared > 0.3


class TestHospitalMeds:
    def test_interaction_model(self):
        df = load_rda("CountyHealth")
        data = df.with_columns([
            pl.col("MDs").sqrt().alias("sqrtMDs"),
            (pl.col("Hospitals") * pl.col("Beds")).alias("Hosp_Beds"),
        ])
        model = fit_lm(data, "sqrtMDs ~ Hospitals + Beds + Hosp_Beds")
        assert model.rsquared > 0.8


class TestCustomerSpending:
    def test_avgspent_model(self):
        df = load_rda("Clothing")
        data = df.filter((pl.col("Amount") > 0) & (pl.col("Amount") < 1_506_000))
        cdata = data.filter(pl.col("Dollar12") > 0).with_columns(
            (pl.col("Dollar12") / pl.col("Freq12")).alias("AvgSpent12")
        )
        model = fit_lm(cdata, "Amount ~ AvgSpent12")
        assert model.rsquared > 0.8  # ~0.909


class TestBrainpH:
    def test_gender_groups(self):
        df = load_rda("BrainpH")
        sexes = df["Sex"].unique().to_list()
        assert len(sexes) >= 2


class TestHandwriting:
    def test_indicator_model(self):
        df = load_rda("Handwriting")
        model = fit_lm(df, "Survey2 ~ Survey1 + Gender")
        assert model.rsquared > 0.1