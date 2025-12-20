"""
Pytest configuration and shared fixtures for DIANA tests.

This module provides common fixtures used across all test modules,
including temporary directories, dummy data paths, and cleanup utilities.

Note: The diana package must be installed in development mode first:
    pip install -e .
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def dummy_matrix_path(fixtures_dir):
    """Return path to dummy matrix file."""
    return fixtures_dir / "dummy_data.pa.mat"


@pytest.fixture
def dummy_metadata_path(fixtures_dir):
    """Return path to dummy metadata file."""
    return fixtures_dir / "dummy_metadata.tsv"


@pytest.fixture
def dummy_config_path(fixtures_dir):
    """Return path to dummy config file."""
    return fixtures_dir / "test_config.yaml"
