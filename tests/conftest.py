"""
tests/conftest.py

Pytest configuration and shared fixtures for the vision_detect_segment test suite.

Slow tests are tests that load deep learning models and are therefore time-consuming.
They are skipped by default and can be enabled with: pytest --run-slow
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options to pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that load deep learning models.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (loads deep learning models). " "Deselected by default; use --run-slow to include them.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip slow tests unless --run-slow flag is provided."""
    if config.getoption("--run-slow"):
        return  # Don't skip anything

    skip_slow = pytest.mark.skip(reason="Slow test skipped by default. Use --run-slow to enable.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
