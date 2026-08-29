"""Tests for UnitB – ANOVA (ch05, ch06, ch07)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from stat2lib.data import load_rda
from stat2lib.stats import fit_lm, anova_table, one_way_anova, levene_test


class TestFatRats:
    def test_oneway_anova(self):
        df = load_rda("FatRats")
        high = df.filter(pl.col("Protein") == "Hi")
        sources = high["Source"].unique().to_list()
        groups = [high.filter(pl.col("Source") == s)["Gain"].to_numpy() for s in sources]
        result = one_way_anova(*groups)
        assert result["pvalue"] < 0.1  # Julia: ~0.0503


class TestTeenPregnancy:
    def test_oneway_anova(self):
        df = load_rda("TeenPregnancy")
        cats = df["CivilWar"].unique().to_list()
        groups = [df.filter(pl.col("CivilWar") == c)["Teen"].to_numpy() for c in cats]
        result = one_way_anova(*groups)
        assert result["pvalue"] < 0.001  # Julia: ~0.0002


class TestWalkingBabies:
    def test_oneway_anova(self):
        df = load_rda("WalkingBabies")
        cats = df["Group"].unique().to_list()
        groups = [df.filter(pl.col("Group") == c)["Age"].to_numpy() for c in cats]
        result = one_way_anova(*groups)
        assert result["pvalue"] > 0.05  # Julia: ~0.1039

    def test_levene(self):
        df = load_rda("WalkingBabies")
        cats = df["Group"].unique().to_list()
        groups = [df.filter(pl.col("Group") == c)["Age"].to_numpy() for c in cats]
        result = levene_test(*groups)
        assert result["pvalue"] > 0.5  # Julia: ~0.6717

    def test_anova_lm(self):
        df = load_rda("WalkingBabies")
        model = fit_lm(df, "Age ~ Group")
        aov = anova_table(model)
        assert aov.shape[0] >= 2


class TestMetroCommutes:
    def test_levene_significant(self):
        df = load_rda("MetroCommutes")
        cats = df["City"].unique().to_list()
        groups = [df.filter(pl.col("City") == c)["Time"].to_numpy() for c in cats]
        result = levene_test(*groups)
        assert result["pvalue"] < 0.001  # Julia: <1e-5


class TestLeafhoppers:
    def test_anova(self):
        df = load_rda("Leafhoppers")
        model = fit_lm(df, "Days ~ Diet")
        aov = anova_table(model)
        assert aov.shape[0] >= 2


class TestFruitFlies:
    def test_anova(self):
        df = load_rda("FruitFlies")
        model = fit_lm(df, "Longevity ~ Treatment")
        aov = anova_table(model)
        assert aov.shape[0] >= 2


class TestRadioactiveTwins:
    def test_oneway(self):
        df = load_rda("RadioactiveTwins")
        envs = df["Env"].unique().to_list()
        groups = [df.filter(pl.col("Env") == e)["Rate"].to_numpy() for e in envs]
        result = one_way_anova(*groups)
        assert result["pvalue"] > 0.5  # Julia: ~0.7037


class TestSleepingShrews:
    def test_oneway(self):
        df = load_rda("SleepingShrews")
        phases = df["Phase"].unique().to_list()
        groups = [df.filter(pl.col("Phase") == p)["Rate"].to_numpy() for p in phases]
        result = one_way_anova(*groups)
        assert result["pvalue"] > 0.5  # Julia: ~0.5973


class TestFranticFingers:
    def test_twoway_anova(self):
        df = load_rda("FranticFingers")
        model = fit_lm(df, "Rate ~ Drug + Subj")
        aov = anova_table(model)
        assert aov.shape[0] >= 3  # Drug + Subj + Residuals


class TestPigFeed:
    def test_twoway_interaction(self):
        df = load_rda("PigFeed")
        model = fit_lm(df, "WgtGain ~ Antibiotic * B12")
        aov = anova_table(model)
        assert aov.shape[0] >= 4


class TestBirdCalcium:
    def test_twoway(self):
        df = load_rda("BirdCalcium")
        model = fit_lm(df, "Ca ~ Hormone * Sex")
        aov = anova_table(model)
        assert aov.shape[0] >= 4

    def test_log_twoway(self):
        df = load_rda("BirdCalcium")
        df = df.with_columns(pl.col("Ca").log10().alias("logCa"))
        model = fit_lm(df, "logCa ~ Hormone * Sex")
        aov = anova_table(model)
        assert aov.shape[0] >= 4