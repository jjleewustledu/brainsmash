"""
Pytest configuration and shared fixtures for brainsmash tests.

Provides synthetic test data, reference values, and utilities for testing
the Python 3.13 port.
"""
import numpy as np
import pytest
import tempfile
import os


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def small_brain_map():
    """Generate a small synthetic brain map (100 elements) with known seed."""
    rng = np.random.RandomState(42)
    return rng.randn(100)


@pytest.fixture
def small_brain_map_with_nans():
    """Generate a brain map with some NaN values (masked regions)."""
    rng = np.random.RandomState(42)
    brain_map = rng.randn(100)
    brain_map[0:5] = np.nan  # Mask first 5 elements
    return brain_map


@pytest.fixture
def small_distance_matrix():
    """
    Generate a small symmetric distance matrix (100x100).

    Uses Euclidean distances between random 2D coordinates to ensure
    the matrix has realistic distance properties (symmetric, zero diagonal,
    triangle inequality satisfied).
    """
    rng = np.random.RandomState(42)
    coords = rng.randn(100, 2) * 10  # 100 points in 2D

    # Compute pairwise Euclidean distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    return D


@pytest.fixture
def sorted_distance_matrix(small_distance_matrix):
    """
    Return sorted distance matrix and corresponding indices.

    Each row is sorted in ascending order, as expected by the Sampled class.
    """
    D = small_distance_matrix
    index = np.argsort(D, axis=1)
    D_sorted = np.take_along_axis(D, index, axis=1)
    return D_sorted, index


@pytest.fixture
def distance_arrays_1d():
    """Generate 1D distance arrays for kernel testing."""
    return np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


@pytest.fixture
def distance_arrays_2d():
    """Generate 2D distance arrays for kernel testing."""
    rng = np.random.RandomState(42)
    return rng.rand(10, 20) * 10


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_brain_map_file(temp_dir, small_brain_map):
    """Write brain map to temporary text file."""
    filepath = os.path.join(temp_dir, "brain_map.txt")
    np.savetxt(filepath, small_brain_map)
    return filepath


@pytest.fixture
def temp_distance_matrix_file(temp_dir, small_distance_matrix):
    """Write distance matrix to temporary text file."""
    filepath = os.path.join(temp_dir, "distmat.txt")
    np.savetxt(filepath, small_distance_matrix, delimiter=' ')
    return filepath


@pytest.fixture
def temp_npy_file(temp_dir, small_brain_map):
    """Write brain map to temporary npy file."""
    filepath = os.path.join(temp_dir, "brain_map.npy")
    np.save(filepath, small_brain_map)
    return filepath


# ============================================================================
# Reference Values for Numerical Accuracy Tests
# ============================================================================

@pytest.fixture
def reference_kernel_values():
    """
    Pre-computed reference values for kernel functions.

    These values can be used to verify numerical accuracy across Python versions.
    """
    d = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    d_max = d.max()

    return {
        'input': d,
        'gaussian': np.exp(-1.25 * np.square(d / d_max)),
        'exp': np.exp(-d / d_max),
        'invdist': 1.0 / d,
        'uniform': np.ones(d.size) / d.size,
    }


@pytest.fixture
def reference_variogram_params():
    """
    Parameters for variogram computation reference tests.
    """
    return {
        'nh': 25,
        'pv': 25,
        'b_multiplier': 3.0,
    }


# ============================================================================
# Statistical Reference Fixtures
# ============================================================================

@pytest.fixture
def correlation_test_data():
    """Generate test data for correlation function testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(5, 100)  # 5 samples, 100 features
    Y = rng.randn(3, 100)  # 3 samples, 100 features
    return X, Y


@pytest.fixture
def perfectly_correlated_data():
    """Data with known perfect correlation for validation."""
    rng = np.random.RandomState(42)
    X = rng.randn(1, 100)
    Y = X.copy()  # r = 1.0
    Z = -X.copy()  # r = -1.0
    return X, Y, Z


# ============================================================================
# Tolerances for Numerical Comparisons
# ============================================================================

@pytest.fixture
def numerical_tolerances():
    """
    Standard tolerances for numerical comparisons.

    Different tolerances for different types of computations to account
    for floating-point accumulation differences across Python versions.
    """
    return {
        'kernel_exact': {'atol': 1e-12, 'rtol': 0},
        'variogram': {'atol': 1e-10, 'rtol': 1e-6},
        'correlation': {'atol': 1e-10, 'rtol': 1e-6},
        'surrogate_seeded': {'atol': 1e-10, 'rtol': 0},
        'statistical': {'atol': 1e-6, 'rtol': 1e-4},
    }


# ============================================================================
# Helper Functions
# ============================================================================

def assert_array_almost_equal(actual, expected, tol_dict):
    """Helper function for array comparison with tolerances."""
    np.testing.assert_allclose(
        actual, expected,
        atol=tol_dict.get('atol', 1e-10),
        rtol=tol_dict.get('rtol', 1e-6)
    )


def is_symmetric(matrix, atol=1e-10):
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T, atol=atol)


def is_valid_distance_matrix(D, atol=1e-10):
    """Check if D is a valid distance matrix (symmetric, zero diagonal, non-negative)."""
    if not is_symmetric(D, atol):
        return False
    if not np.allclose(np.diag(D), 0, atol=atol):
        return False
    if np.any(D < -atol):
        return False
    return True
