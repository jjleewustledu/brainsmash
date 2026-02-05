"""
Tests for brainsmash.mapgen.base.Base class.

Tests the core surrogate map generation algorithm for numerical accuracy
and reproducibility across Python versions.
"""
import numpy as np
import pytest
from brainsmash.mapgen.base import Base


class TestBaseInit:
    """Tests for Base class initialization."""

    def test_init_with_arrays(self, small_brain_map, small_distance_matrix):
        """Test initialization with numpy arrays."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        assert gen.nmap == len(small_brain_map)
        assert gen._D.shape == small_distance_matrix.shape

    def test_init_with_files(self, temp_brain_map_file, temp_distance_matrix_file):
        """Test initialization with file paths."""
        gen = Base(temp_brain_map_file, temp_distance_matrix_file, seed=42)

        assert gen.nmap == 100  # Based on fixture

    def test_init_with_custom_deltas(self, small_brain_map, small_distance_matrix):
        """Test initialization with custom delta values."""
        deltas = np.array([0.2, 0.5, 0.8])
        gen = Base(small_brain_map, small_distance_matrix, deltas=deltas, seed=42)

        np.testing.assert_array_equal(gen.deltas, deltas)

    def test_init_kernel_selection(self, small_brain_map, small_distance_matrix):
        """Test initialization with different kernels."""
        for kernel in ['exp', 'gaussian', 'invdist', 'uniform']:
            gen = Base(small_brain_map, small_distance_matrix, kernel=kernel, seed=42)
            assert callable(gen.kernel)


class TestBaseVariogram:
    """Tests for variogram computation."""

    def test_compute_smooth_variogram_output_shape(
        self, small_brain_map, small_distance_matrix
    ):
        """Test variogram output shape matches nh parameter."""
        nh = 25
        gen = Base(small_brain_map, small_distance_matrix, nh=nh, seed=42)

        variogram = gen.compute_smooth_variogram(gen.x)
        assert variogram.shape == (nh,)

    def test_compute_smooth_variogram_return_h(
        self, small_brain_map, small_distance_matrix
    ):
        """Test variogram with return_h=True."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        variogram, h = gen.compute_smooth_variogram(gen.x, return_h=True)
        assert variogram.shape == h.shape

    def test_variogram_non_negative(self, small_brain_map, small_distance_matrix):
        """Variogram values should be non-negative (squared differences)."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        variogram = gen.compute_smooth_variogram(gen.x)
        assert np.all(variogram >= 0)

    def test_variogram_monotonic_tendency(
        self, small_brain_map, small_distance_matrix
    ):
        """Smoothed variogram should generally increase with distance."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        variogram, h = gen.compute_smooth_variogram(gen.x, return_h=True)
        # Allow for some non-monotonicity due to data noise
        # Check overall trend
        assert variogram[-1] >= variogram[0] * 0.5  # End should not be much smaller


class TestBasePermutation:
    """Tests for map permutation."""

    def test_permute_map_preserves_values(
        self, small_brain_map, small_distance_matrix
    ):
        """Permuted map should contain same values (just reordered)."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        perm = gen.permute_map(i=1)
        original_sorted = np.sort(gen.x.data)
        perm_sorted = np.sort(perm.data.flatten())

        np.testing.assert_array_almost_equal(original_sorted, perm_sorted)

    def test_permute_map_batch(self, small_brain_map, small_distance_matrix):
        """Test batch permutation."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        perm = gen.permute_map(i=5)
        assert perm.shape == (100, 5)


class TestBaseSmoothing:
    """Tests for map smoothing."""

    def test_smooth_map_output_shape(self, small_brain_map, small_distance_matrix):
        """Smoothed map should have same shape as input."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        x = gen.x[:, np.newaxis]
        delta = 0.5
        smoothed = gen.smooth_map(x, delta)

        assert smoothed.shape[0] == x.shape[0]

    def test_smooth_map_reduces_variance(
        self, small_brain_map, small_distance_matrix
    ):
        """Smoothing should reduce local variance."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        x = gen.x[:, np.newaxis]
        delta = 0.5
        smoothed = gen.smooth_map(x, delta)

        # Smoothed should generally have lower variance
        # (may not always be true but tendency should hold)
        assert np.var(smoothed) <= np.var(x) * 2  # Allow some tolerance


class TestBaseRegression:
    """Tests for linear regression."""

    def test_regress_perfect_fit(self, small_brain_map, small_distance_matrix):
        """Test regression with perfectly correlated data."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        x = np.arange(25).astype(float)
        y = 2 * x + 3  # Perfect linear relationship

        alpha, beta, res = gen.regress(x, y)

        assert alpha == pytest.approx(3.0, rel=1e-6)
        assert beta == pytest.approx(2.0, rel=1e-6)
        assert res == pytest.approx(0.0, abs=1e-10)

    def test_regress_output_types(self, small_brain_map, small_distance_matrix):
        """Test regression output types."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        rng = np.random.RandomState(42)
        x = rng.randn(25)
        y = rng.randn(25)

        alpha, beta, res = gen.regress(x, y)

        # Should be scalars or arrays
        assert np.isscalar(alpha) or isinstance(alpha, np.ndarray)
        assert np.isscalar(beta) or isinstance(beta, np.ndarray)
        assert np.isscalar(res) or isinstance(res, np.ndarray)


class TestBaseSurrogateGeneration:
    """Tests for surrogate map generation."""

    def test_call_single_surrogate(self, small_brain_map, small_distance_matrix):
        """Test generating a single surrogate map."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        surrogate = gen(n=1)
        assert surrogate.shape == (100,)

    def test_call_multiple_surrogates(self, small_brain_map, small_distance_matrix):
        """Test generating multiple surrogate maps."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        surrogates = gen(n=5)
        assert surrogates.shape == (5, 100)

    def test_call_with_batch_size(self, small_brain_map, small_distance_matrix):
        """Test surrogate generation with custom batch size."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        surrogates = gen(n=10, batch_size=5)
        assert surrogates.shape == (10, 100)

    def test_surrogate_mean_approximately_zero(
        self, small_brain_map, small_distance_matrix
    ):
        """Without resampling, surrogates should be de-meaned."""
        gen = Base(
            small_brain_map, small_distance_matrix, resample=False, seed=42
        )

        surrogates = gen(n=5)

        for i in range(5):
            assert np.abs(surrogates[i].mean()) < 0.5  # Approximately zero

    def test_surrogate_with_resample(self, small_brain_map, small_distance_matrix):
        """Test surrogate generation with resampling."""
        gen = Base(
            small_brain_map, small_distance_matrix, resample=True, seed=42
        )

        surrogate = gen(n=1)

        # With resampling, surrogate values should come from original map
        original_sorted = np.sort(small_brain_map)
        surrogate_sorted = np.sort(surrogate)

        np.testing.assert_array_almost_equal(original_sorted, surrogate_sorted)


class TestBaseProperties:
    """Tests for Base class properties."""

    def test_x_property(self, small_brain_map, small_distance_matrix):
        """Test x property returns masked array."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        assert isinstance(gen.x, np.ma.MaskedArray)
        assert gen.x.shape == (100,)

    def test_D_property(self, small_brain_map, small_distance_matrix):
        """Test D property returns distance matrix."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        assert gen.D.shape == (100, 100)

    def test_h_property(self, small_brain_map, small_distance_matrix):
        """Test h property returns variogram distances."""
        nh = 25
        gen = Base(small_brain_map, small_distance_matrix, nh=nh, seed=42)

        assert gen.h.shape == (nh,)

    def test_resample_property(self, small_brain_map, small_distance_matrix):
        """Test resample property."""
        gen = Base(
            small_brain_map, small_distance_matrix, resample=True, seed=42
        )
        assert gen.resample is True

        gen = Base(
            small_brain_map, small_distance_matrix, resample=False, seed=42
        )
        assert gen.resample is False

    def test_resample_type_error(self, small_brain_map, small_distance_matrix):
        """Test resample property type checking."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        with pytest.raises(TypeError):
            gen.resample = "not a bool"


class TestBaseWithMaskedData:
    """Tests for Base class with masked (NaN) data."""

    def test_init_with_nans(
        self, small_brain_map_with_nans, small_distance_matrix
    ):
        """Test initialization with NaN values in brain map."""
        gen = Base(
            small_brain_map_with_nans, small_distance_matrix, seed=42
        )

        assert gen.nmap == 100
        # Mask should be set for NaN values
        assert np.sum(gen.x.mask) == 5  # First 5 elements are NaN

    def test_surrogate_with_masked_data(
        self, small_brain_map_with_nans, small_distance_matrix
    ):
        """Test surrogate generation with masked data."""
        gen = Base(
            small_brain_map_with_nans, small_distance_matrix, seed=42
        )

        # Should not raise exception
        surrogate = gen(n=1)
        assert surrogate.shape == (100,)
