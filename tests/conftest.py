"""Shared fixtures for Memory Maze Genesis backend tests."""

import numpy as np
import pytest

# Skip entire test module if Genesis is not installed
try:
    import genesis as gs
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False

pytestmark = pytest.mark.skipif(not HAS_GENESIS, reason="Genesis not installed")


@pytest.fixture(scope="session")
def init_genesis():
    """Initialize Genesis once per test session."""
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    return gs


@pytest.fixture
def rng():
    """Deterministic RNG for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def rng_factory():
    """Factory that creates independent RNGs from sequential seeds."""
    _counter = [0]

    def _make(seed=None):
        if seed is None:
            seed = 1000 + _counter[0]
            _counter[0] += 1
        return np.random.RandomState(seed)

    return _make
