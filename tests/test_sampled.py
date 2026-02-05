"""
Tests for brainsmash.mapgen.sampled.Sampled class.

Tests the memory-efficient surrogate map generation algorithm.
"""
import numpy as np
import pytest
from brainsmash.mapgen.sampled import Sampled


class TestSampledInit:
    """Tests for Sampled class initialization."""

    def test_init_with_arrays(self, sorted_distance_matrix, small_brain_map):
        """Test initialization with numpy arrays."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        assert gen.nmap == len(small_brain_map)

    def test_init_with_custom_params(self, sorted_distance_matrix, small_brain_map):
        """Test initialization with custom parameters."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=30,
            knn=40,
            nh=20,
            pv=60,
            seed=42
        )

        assert gen.ns == 30
        assert gen.knn == 40
        assert gen.nh == 20
        assert gen.pv == 60

    def test_init_kernel_selection(self, sorted_distance_matrix, small_brain_map):
        """Test initialization with different kernels."""
        D_sorted, index = sorted_distance_matrix

        for kernel in ['exp', 'gaussian', 'invdist', 'uniform']:
            gen = Sampled(
                small_brain_map,
                D_sorted,
                index,
                ns=50,
                knn=50,
                kernel=kernel,
                seed=42
            )
            assert callable(gen.kernel)


class TestSampledVariogram:
    """Tests for variogram computation in Sampled class."""

    def test_compute_variogram_output_shape(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Test variogram computation output shape."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        idx = gen.sample()
        v = gen.compute_variogram(gen._x, idx)

        # Output shape should be (ns, knn)
        assert v.shape[0] == len(idx)

    def test_variogram_non_negative(self, sorted_distance_matrix, small_brain_map):
        """Variogram values should be non-negative."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        idx = gen.sample()
        v = gen.compute_variogram(gen._x, idx)

        assert np.all(v >= 0)


class TestSampledSmoothing:
    """Tests for map smoothing in Sampled class."""

    def test_smooth_variogram_output_shape(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Test smooth_variogram output shape."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            nh=25,
            seed=42
        )

        u = np.random.rand(100)
        v = np.random.rand(100)

        smoothed = gen.smooth_variogram(u, v)
        assert smoothed.shape == (25,)

    def test_smooth_variogram_return_h(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Test smooth_variogram with return_h=True."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            nh=25,
            seed=42
        )

        u = np.random.rand(100)
        v = np.random.rand(100)

        smoothed, h = gen.smooth_variogram(u, v, return_h=True)
        assert smoothed.shape == h.shape

    def test_smooth_variogram_size_mismatch(
        self, sorted_distance_matrix, small_brain_map
    ):
        """smooth_variogram should raise ValueError for mismatched sizes."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        u = np.random.rand(100)
        v = np.random.rand(50)  # Different size

        with pytest.raises(ValueError):
            gen.smooth_variogram(u, v)


class TestSampledPermutation:
    """Tests for map permutation in Sampled class."""

    def test_permute_map_preserves_values(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Permuted map should contain same values."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        perm = gen.permute_map()
        original_sorted = np.sort(gen._x)
        perm_sorted = np.sort(perm)

        np.testing.assert_array_almost_equal(original_sorted, perm_sorted)


class TestSampledSampling:
    """Tests for random sampling in Sampled class."""

    def test_sample_output_shape(self, sorted_distance_matrix, small_brain_map):
        """Test sample() output shape."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        idx = gen.sample()
        assert idx.shape == (50,)

    def test_sample_no_replacement(self, sorted_distance_matrix, small_brain_map):
        """Sampled indices should be unique (no replacement)."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        idx = gen.sample()
        assert len(np.unique(idx)) == len(idx)

    def test_sample_in_range(self, sorted_distance_matrix, small_brain_map):
        """Sampled indices should be valid indices."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        idx = gen.sample()
        assert np.all(idx >= 0)
        assert np.all(idx < gen.nmap)


class TestSampledRegression:
    """Tests for linear regression in Sampled class."""

    def test_regress_perfect_fit(self, sorted_distance_matrix, small_brain_map):
        """Test regression with perfectly correlated data."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        x = np.arange(25).astype(float)
        y = 2 * x + 3

        alpha, beta, res = gen.regress(x, y)

        assert alpha == pytest.approx(3.0, rel=1e-4)
        assert beta == pytest.approx(2.0, rel=1e-4)
        assert res == pytest.approx(0.0, abs=1e-6)


class TestSampledSurrogateGeneration:
    """Tests for surrogate map generation."""

    def test_call_single_surrogate(self, sorted_distance_matrix, small_brain_map):
        """Test generating a single surrogate map."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        surrogate = gen(n=1)
        assert surrogate.shape == (100,)

    def test_call_multiple_surrogates(self, sorted_distance_matrix, small_brain_map):
        """Test generating multiple surrogate maps."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        surrogates = gen(n=3)
        assert surrogates.shape == (3, 100)

    def test_surrogate_mean_approximately_zero(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Without resampling, surrogates should be de-meaned."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            resample=False,
            seed=42
        )

        surrogates = gen(n=3)

        for i in range(3):
            # Mean should be approximately zero (allowing for numerical precision)
            assert np.abs(np.nanmean(surrogates[i])) < 1.0

    def test_surrogate_with_resample(self, sorted_distance_matrix, small_brain_map):
        """Test surrogate generation with resampling."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            resample=True,
            seed=42
        )

        surrogate = gen(n=1)

        # With resampling, surrogate values should come from original map
        original_sorted = np.sort(small_brain_map)
        surrogate_sorted = np.sort(surrogate)

        np.testing.assert_array_almost_equal(original_sorted, surrogate_sorted)


class TestSampledProperties:
    """Tests for Sampled class properties."""

    def test_x_property(self, sorted_distance_matrix, small_brain_map):
        """Test x property returns copy of brain map."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        x = gen.x
        assert x.shape == (100,)

    def test_D_property(self, sorted_distance_matrix, small_brain_map):
        """Test D property returns truncated distance matrix."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        D = gen.D
        # D is truncated to knn columns (excluding self)
        assert D.shape == (100, 50)

    def test_resample_property(self, sorted_distance_matrix, small_brain_map):
        """Test resample property."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            resample=True,
            seed=42
        )
        assert gen.resample is True

    def test_resample_type_error(self, sorted_distance_matrix, small_brain_map):
        """Test resample property type checking."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        with pytest.raises(TypeError):
            gen.resample = "not a bool"


class TestSampledWithMaskedData:
    """Tests for Sampled class with masked (NaN) data."""

    def test_init_with_nans(
        self, sorted_distance_matrix, small_brain_map_with_nans
    ):
        """Test initialization with NaN values in brain map."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map_with_nans,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        assert gen.nmap == 100
        # Should detect masked status
        assert gen._ismasked is True

    def test_surrogate_with_masked_data(
        self, sorted_distance_matrix, small_brain_map_with_nans
    ):
        """Test surrogate generation with masked data."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map_with_nans,
            D_sorted,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        # Should not raise exception
        surrogate = gen(n=1)
        assert surrogate.shape == (100,)
