"""Shared fixtures. All tests run offline — no network, no live downloads (CI-safe)."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture()
def sample_events_csv() -> Path:
    """Synthetic GOES event list (year 2099 — deliberately not real data)."""
    return FIXTURES_DIR / "goes_events_sample.csv"


@pytest.fixture()
def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "default.yaml"
