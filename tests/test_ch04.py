"""Tests for ch04 – Advanced Regression Topics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm


class TestAddedVariablePlot:
    def test_price_beds(self):
        df = load_rda("HousesNY")
        data = df.select(["Beds", "Size", "Price"])
        model = fit_lm(data, "Price ~ Beds")
        assert model.rsquared > 0.1  # ~0.175

    def test_size_beds(self):
        df = load_rda("HousesNY")
        data = df.select(["Beds", "Size", "Price"])
        model = fit_lm(data, "Size ~ Beds")
        assert model.rsquared > 0.5  # strong relationship


class TestFirstYearGPA:
    def test_data_shape(self):
        df = load_rda("FirstYearGPA")
        assert "GPA" in df.columns
        assert "HSGPA" in df.columns
        assert df.shape[0] > 100


class TestThreeCars2017:
    def test_onehot(self):
        df = load_rda("ThreeCars2017")
        dummies = df.select(["CarType"]).to_dummies("CarType")
        assert dummies.shape[1] >= 3  # Mazda6, Accord, Maxima


class TestAccordPricePredict:
    def test_model(self):
        df = load_rda("AccordPrice")
        data = df.select(["Mileage", "Price"])
        model = fit_lm(data, "Price ~ Mileage")
        assert model.params["Mileage"] < 0
        assert model.rsquared > 0.7