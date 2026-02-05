"""
Tests for brainsmash.mapgen.stats module.

Tests statistical functions for numerical accuracy against scipy references.
"""
import numpy as np
import pytest
from scipy import stats as scipy_stats
from brainsmash.mapgen.stats import pearsonr, spearmanr, pairwise_r, nonparp


class TestPearsonr:
    """Tests for multi-dimensional Pearson correlation."""

    def test_pearsonr_basic(self, correlation_test_data, numerical_tolerances):
        """Test basic Pearson correlation computation."""
        X, Y = correlation_test_data
        result = pearsonr(X, Y)

        # Result should be (N, M) where N=X.shape[0], M=Y.shape[0]
        assert result.shape == (X.shape[0], Y.shape[0])

    def test_pearsonr_against_scipy(self, numerical_tolerances):
        """Verify Pearson correlation against scipy.stats.pearsonr."""
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)

        # Our implementation (expects 2D input)
        result = pearsonr(x.reshape(1, -1), y.reshape(1, -1))

        # scipy reference
        expected, _ = scipy_stats.pearsonr(x, y)

        np.testing.assert_allclose(
            result[0, 0], expected, **numerical_tolerances['correlation']
        )

    def test_pearsonr_perfect_positive(self, perfectly_correlated_data, numerical_tolerances):
        """Test Pearson correlation with perfect positive correlation."""
        X, Y, _ = perfectly_correlated_data
        result = pearsonr(X, Y)
        np.testing.assert_allclose(result[0, 0], 1.0, **numerical_tolerances['correlation'])

    def test_pearsonr_perfect_negative(self, perfectly_correlated_data, numerical_tolerances):
        """Test Pearson correlation with perfect negative correlation."""
        X, _, Z = perfectly_correlated_data
        result = pearsonr(X, Z)
        np.testing.assert_allclose(result[0, 0], -1.0, **numerical_tolerances['correlation'])

    def test_pearsonr_1d_input(self, numerical_tolerances):
        """Test that 1D input is handled correctly (reshaped to 2D)."""
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)

        result = pearsonr(x, y)
        assert result.shape == (1, 1)

    def test_pearsonr_type_error(self):
        """pearsonr should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            pearsonr([1, 2, 3], np.array([1, 2, 3]))

    def test_pearsonr_size_mismatch(self):
        """pearsonr should raise ValueError for mismatched sizes."""
        X = np.random.randn(5, 100)
        Y = np.random.randn(3, 50)  # Different number of features
        with pytest.raises(ValueError):
            pearsonr(X, Y)


class TestSpearmanr:
    """Tests for multi-dimensional Spearman rank correlation."""

    def test_spearmanr_basic(self, correlation_test_data, numerical_tolerances):
        """Test basic Spearman correlation computation."""
        X, Y = correlation_test_data
        result = spearmanr(X, Y)

        assert result.shape == (X.shape[0], Y.shape[0])

    def test_spearmanr_against_scipy(self, numerical_tolerances):
        """Verify Spearman correlation against scipy.stats.spearmanr."""
        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)

        # Our implementation
        result = spearmanr(x.reshape(1, -1), y.reshape(1, -1))

        # scipy reference
        expected, _ = scipy_stats.spearmanr(x, y)

        np.testing.assert_allclose(
            result[0, 0], expected, **numerical_tolerances['correlation']
        )

    def test_spearmanr_perfect_positive(self, numerical_tolerances):
        """Test Spearman correlation with perfect monotonic relationship."""
        x = np.arange(100).reshape(1, -1).astype(float)
        y = x.copy()
        result = spearmanr(x, y)
        np.testing.assert_allclose(result[0, 0], 1.0, **numerical_tolerances['correlation'])

    def test_spearmanr_perfect_negative(self, numerical_tolerances):
        """Test Spearman correlation with perfect negative monotonic relationship."""
        x = np.arange(100).reshape(1, -1).astype(float)
        y = -x.copy()
        result = spearmanr(x, y)
        np.testing.assert_allclose(result[0, 0], -1.0, **numerical_tolerances['correlation'])

    def test_spearmanr_type_error(self):
        """spearmanr should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            spearmanr([1, 2, 3], np.array([1, 2, 3]))


class TestPairwiseR:
    """Tests for pairwise Pearson correlation function."""

    def test_pairwise_r_shape(self, correlation_test_data):
        """Test output shape of pairwise_r."""
        X, _ = correlation_test_data  # X is (5, 100)
        result = pairwise_r(X)

        assert result.shape == (X.shape[0], X.shape[0])

    def test_pairwise_r_symmetric(self, correlation_test_data, numerical_tolerances):
        """pairwise_r should produce symmetric matrix."""
        X, _ = correlation_test_data
        result = pairwise_r(X)

        np.testing.assert_allclose(
            result, result.T, **numerical_tolerances['correlation']
        )

    def test_pairwise_r_diagonal(self, correlation_test_data, numerical_tolerances):
        """Diagonal of pairwise_r should be 1.0 (self-correlation)."""
        X, _ = correlation_test_data
        result = pairwise_r(X)

        np.testing.assert_allclose(
            np.diag(result), 1.0, **numerical_tolerances['correlation']
        )

    def test_pairwise_r_flatten(self, correlation_test_data):
        """Test flattened output mode."""
        X, _ = correlation_test_data  # X is (5, 100)
        result_flat = pairwise_r(X, flatten=True)

        # Upper triangular has n*(n-1)/2 elements
        n = X.shape[0]
        expected_len = n * (n - 1) // 2

        assert result_flat.shape == (expected_len,)


class TestNonparp:
    """Tests for non-parametric p-value computation."""

    def test_nonparp_basic(self):
        """Test basic non-parametric p-value computation."""
        stat = 2.0
        dist = np.array([-3.0, -2.5, -1.0, 0.0, 1.0, 2.5, 3.0])

        result = nonparp(stat, dist)

        # Count values with |x| > |stat|=2.0: -3.0, -2.5, 2.5, 3.0 = 4 out of 7
        expected = 4 / 7
        assert result == pytest.approx(expected)

    def test_nonparp_extreme_stat(self):
        """Test p-value when stat is more extreme than all null values."""
        stat = 10.0
        dist = np.array([-1.0, 0.0, 1.0, 2.0])

        result = nonparp(stat, dist)
        assert result == 0.0

    def test_nonparp_stat_in_middle(self):
        """Test p-value when stat is in the middle of distribution."""
        stat = 0.5
        dist = np.linspace(-5, 5, 1001)

        result = nonparp(stat, dist)
        # About 90% of values should have |x| > 0.5
        assert 0.85 < result < 0.95

    def test_nonparp_negative_stat(self):
        """Test that negative stat is handled correctly (uses abs)."""
        dist = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        result_pos = nonparp(2.0, dist)
        result_neg = nonparp(-2.0, dist)

        assert result_pos == result_neg


class TestStatisticalNumericalStability:
    """Tests for numerical stability across Python versions."""

    def test_correlation_determinism(self, correlation_test_data):
        """Correlation functions should be deterministic."""
        X, Y = correlation_test_data

        result1 = pearsonr(X.copy(), Y.copy())
        result2 = pearsonr(X.copy(), Y.copy())
        np.testing.assert_array_equal(result1, result2)

        result1 = spearmanr(X.copy(), Y.copy())
        result2 = spearmanr(X.copy(), Y.copy())
        np.testing.assert_array_equal(result1, result2)

    def test_correlation_with_constant(self):
        """Test correlation when one variable is constant (edge case)."""
        X = np.ones((1, 100))
        Y = np.random.randn(1, 100)

        # This may produce NaN due to zero variance - that's expected behavior
        result = pearsonr(X, Y)
        # Just verify it doesn't crash
        assert result.shape == (1, 1)

    def test_pairwise_r_single_row(self):
        """Test pairwise_r with single row (edge case)."""
        X = np.random.randn(1, 100)
        result = pairwise_r(X)

        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)
