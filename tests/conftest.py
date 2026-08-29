"""Shared pytest fixtures for Stat2 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def stat2data_dir() -> Path:
    d = Path(__file__).resolve().parents[1] / ".." / "Stat2Data"
    if not d.is_dir():
        d = Path(__file__).resolve().parents[2] / "Stat2Data"
    if not d.is_dir():
        pytest.skip("Stat2Data directory not found")
    return d.resolve()