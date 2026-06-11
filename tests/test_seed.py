"""Determinism: same seed -> identical RNG draws."""

import random

import numpy as np

from solarflare.utils.seed import set_global_seed


def test_set_global_seed_reproducible():
    set_global_seed(123)
    a_py, a_np = random.random(), np.random.rand(5)
    set_global_seed(123)
    b_py, b_np = random.random(), np.random.rand(5)
    assert a_py == b_py
    assert np.array_equal(a_np, b_np)


def test_different_seeds_differ():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
