"""Smoke tests for the package installation."""

import strand_surfels


def test_version() -> None:
    """The package is importable and exposes a version string."""
    assert isinstance(strand_surfels.__version__, str)
