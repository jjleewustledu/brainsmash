"""
Tests for brainsmash.mapgen.kernels module.

Tests kernel functions for numerical accuracy and edge case handling.
"""
import numpy as np
import pytest
from brainsmash.mapgen.kernels import gaussian, exp, invdist, uniform, check_kernel


class TestGaussianKernel:
    """Tests for the Gaussian kernel function."""

    def test_gaussian_1d_basic(self, distance_arrays_1d, numerical_tolerances):
        """Test Gaussian kernel on 1D array."""
        d = distance_arrays_1d
        result = gaussian(d)

        # Manual calculation
        d_max = d.max()
        expected = np.exp(-1.25 * np.square(d / d_max))

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_gaussian_2d_basic(self, distance_arrays_2d, numerical_tolerances):
        """Test Gaussian kernel on 2D array."""
        d = distance_arrays_2d
        result = gaussian(d)

        # Manual calculation
        d_max = d.max(axis=-1)[:, np.newaxis]
        expected = np.exp(-1.25 * np.square(d / d_max))

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_gaussian_output_range(self, distance_arrays_2d):
        """Gaussian kernel should return values in (0, 1]."""
        result = gaussian(distance_arrays_2d)
        assert np.all(result > 0)
        assert np.all(result <= 1)

    def test_gaussian_max_at_zero(self):
        """Gaussian kernel should be maximal at d=0."""
        d = np.array([0.0, 0.5, 1.0, 2.0])
        result = gaussian(d)
        assert result[0] == pytest.approx(1.0)

    def test_gaussian_type_error(self):
        """Gaussian kernel should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            gaussian("not an array")


class TestExpKernel:
    """Tests for the exponential decay kernel function."""

    def test_exp_1d_basic(self, distance_arrays_1d, numerical_tolerances):
        """Test exponential kernel on 1D array."""
        d = distance_arrays_1d
        result = exp(d)

        d_max = d.max()
        expected = np.exp(-d / d_max)

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_exp_2d_basic(self, distance_arrays_2d, numerical_tolerances):
        """Test exponential kernel on 2D array."""
        d = distance_arrays_2d
        result = exp(d)

        d_max = d.max(axis=-1)[:, np.newaxis]
        expected = np.exp(-d / d_max)

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_exp_output_range(self, distance_arrays_2d):
        """Exponential kernel should return values in (0, 1]."""
        result = exp(distance_arrays_2d)
        assert np.all(result > 0)
        assert np.all(result <= 1)

    def test_exp_max_at_zero(self):
        """Exponential kernel should be maximal at d=0."""
        d = np.array([0.0, 0.5, 1.0, 2.0])
        result = exp(d)
        assert result[0] == pytest.approx(1.0)

    def test_exp_type_error(self):
        """Exponential kernel should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            exp("not an array")


class TestInvdistKernel:
    """Tests for the inverse distance kernel function."""

    def test_invdist_basic(self, numerical_tolerances):
        """Test inverse distance kernel."""
        d = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        result = invdist(d)
        expected = 1.0 / d

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_invdist_2d(self, distance_arrays_2d, numerical_tolerances):
        """Test inverse distance kernel on 2D array."""
        d = distance_arrays_2d
        result = invdist(d)
        expected = 1.0 / d

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_invdist_positive_output(self, distance_arrays_2d):
        """Inverse distance kernel should return positive values for positive d."""
        result = invdist(distance_arrays_2d)
        assert np.all(result > 0)

    def test_invdist_type_error(self):
        """Inverse distance kernel should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            invdist("not an array")


class TestUniformKernel:
    """Tests for the uniform (distance-independent) kernel function."""

    def test_uniform_1d_basic(self, numerical_tolerances):
        """Test uniform kernel on 1D array."""
        d = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        result = uniform(d)
        expected = np.ones(d.size) / d.size

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_uniform_2d_basic(self, distance_arrays_2d, numerical_tolerances):
        """Test uniform kernel on 2D array."""
        d = distance_arrays_2d
        result = uniform(d)
        expected = np.ones(d.shape) / d.shape[-1]

        np.testing.assert_allclose(
            result, expected, **numerical_tolerances['kernel_exact']
        )

    def test_uniform_row_sum(self, distance_arrays_2d):
        """Rows of uniform kernel output should sum to 1."""
        result = uniform(distance_arrays_2d)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_uniform_type_error(self):
        """Uniform kernel should raise TypeError for non-array input."""
        with pytest.raises(TypeError):
            uniform("not an array")


class TestCheckKernel:
    """Tests for kernel validation function."""

    def test_check_kernel_valid(self):
        """check_kernel should return callable for valid kernel names."""
        for name in ['gaussian', 'exp', 'invdist', 'uniform']:
            kernel_func = check_kernel(name)
            assert callable(kernel_func)

    def test_check_kernel_invalid(self):
        """check_kernel should raise NotImplementedError for invalid names."""
        with pytest.raises(NotImplementedError):
            check_kernel('invalid_kernel')

    def test_check_kernel_returns_correct_function(self):
        """check_kernel should return the correct kernel function."""
        assert check_kernel('gaussian') is gaussian
        assert check_kernel('exp') is exp
        assert check_kernel('invdist') is invdist
        assert check_kernel('uniform') is uniform


class TestKernelNumericalStability:
    """Tests for numerical stability across Python versions."""

    def test_kernel_with_very_small_values(self):
        """Test kernels with very small distance values."""
        d = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(gaussian(d)))
        assert np.all(np.isfinite(exp(d)))
        assert np.all(np.isfinite(invdist(d)))
        assert np.all(np.isfinite(uniform(d)))

    def test_kernel_with_large_values(self):
        """Test kernels with large distance values."""
        d = np.array([1e2, 1e4, 1e6, 1e8])

        assert np.all(np.isfinite(gaussian(d)))
        assert np.all(np.isfinite(exp(d)))
        assert np.all(np.isfinite(invdist(d)))
        assert np.all(np.isfinite(uniform(d)))

    def test_kernel_determinism(self, distance_arrays_2d):
        """Kernels should produce identical results on repeated calls."""
        d = distance_arrays_2d

        for kernel_func in [gaussian, exp, invdist, uniform]:
            result1 = kernel_func(d.copy())
            result2 = kernel_func(d.copy())
            np.testing.assert_array_equal(result1, result2)
