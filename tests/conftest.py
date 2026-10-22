"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator so tests are repeatable."""
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A 12-image synthetic dataset in YOLO layout (built once per session)."""
    from rasherqc.synthetic import generate_dataset

    root = tmp_path_factory.mktemp("tiny")
    generate_dataset(root, n_train=6, n_valid=3, n_test=3, seed=7)
    return root
