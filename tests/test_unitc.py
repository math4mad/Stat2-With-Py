"""Tests for UnitC – Logistic Regression (ch09, ch10)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm, fit_glm


class TestLosingSleep:
    def test_logistic_fit(self):
        df = load_rda("LosingSleep")
        model = fit_glm(df, "Outcome ~ Age")
        assert model is not None

    def test_linear_fit(self):
        df = load_rda("LosingSleep")
        ages = sorted(df["Age"].unique().to_list())
        rows = []
        for a in ages:
            sub = df.filter(pl.col("Age") == a)
            n = sub.shape[0]
            m7 = sub.filter(pl.col("Outcome") == 1).shape[0]
            rows.append({"Age": a, "proportion": m7 / n})
        agg = pl.DataFrame(rows)
        model = fit_lm(agg, "proportion ~ Age")
        assert model.params["Age"] < 0  # older → less sleep


class TestMedGPA:
    def test_logistic_significant(self):
        df = load_rda("MedGPA")
        model = fit_glm(df, "Acceptance ~ GPA")
        # GPA should be significant predictor
        assert model.pvalues["GPA"] < 0.01


class TestMigraines:
    def test_data(self):
        df = load_rda("Migraines")
        assert "Group" in df.columns
        assert "Yes" in df.columns
        assert df.shape[0] == 2  # TMS + Placebo


class TestEyes:
    def test_logistic(self):
        df = load_rda("Eyes")
        model = fit_glm(df, "Gay ~ DilateDiff")
        assert model is not None