"""Shared pytest fixtures for the offline TAINT benchmark.

Every fixture is deterministic and offline: no API keys, no downloads, no
network. The benchmark is built once per session and reused.
"""

from __future__ import annotations

import pytest

from taint.dataset import Dataset, build_benchmark


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """A small, balanced, deterministic benchmark shared across tests."""
    return build_benchmark(n=200, test_frac=0.4, seed=0)
