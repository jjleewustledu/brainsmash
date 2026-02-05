"""
Tests for end-to-end integration workflows.

These tests verify complete workflows from data loading through
surrogate generation and statistical analysis.
"""
import numpy as np
import pytest
import os
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.stats import pearsonr, spearmanr, nonparp
from brainsmash.mapgen.memmap import txt2memmap, load_memmap


class TestBaseWorkflow:
    """Integration tests for Base class workflow."""

    def test_full_workflow_arrays(self, small_brain_map, small_distance_matrix):
        """Test complete workflow with numpy arrays."""
        # 1. Create generator
        gen = Base(small_brain_map, small_distance_matrix, seed=42)

        # 2. Generate surrogates
        surrogates = gen(n=10)

        # 3. Compute correlations
        original = small_brain_map.reshape(1, -1)
        corrs = pearsonr(original, surrogates)

        # 4. Compute p-value
        observed_stat = 0.5  # Example statistic
        p_value = nonparp(observed_stat, corrs.flatten())

        # Verify outputs
        assert surrogates.shape == (10, 100)
        assert corrs.shape == (1, 10)
        assert 0 <= p_value <= 1

    def test_full_workflow_files(
        self, temp_brain_map_file, temp_distance_matrix_file
    ):
        """Test complete workflow with file inputs."""
        # 1. Create generator from files
        gen = Base(temp_brain_map_file, temp_distance_matrix_file, seed=42)

        # 2. Generate surrogates
        surrogates = gen(n=5)

        # Verify
        assert surrogates.shape == (5, 100)

    def test_workflow_with_resample(
        self, small_brain_map, small_distance_matrix
    ):
        """Test workflow with resampling enabled."""
        gen = Base(
            small_brain_map, small_distance_matrix,
            resample=True, seed=42
        )

        surrogates = gen(n=5)

        # Verify distribution preservation
        for i in range(5):
            original_sorted = np.sort(small_brain_map)
            surrogate_sorted = np.sort(surrogates[i])
            np.testing.assert_array_almost_equal(
                original_sorted, surrogate_sorted
            )

    def test_workflow_different_kernels(
        self, small_brain_map, small_distance_matrix
    ):
        """Test workflow with different kernel functions."""
        kernels = ['exp', 'gaussian', 'invdist', 'uniform']

        for kernel in kernels:
            gen = Base(
                small_brain_map, small_distance_matrix,
                kernel=kernel, seed=42
            )
            surrogate = gen(n=1)
            assert surrogate.shape == (100,)


class TestSampledWorkflow:
    """Integration tests for Sampled class workflow."""

    def test_full_workflow_memmap(
        self, temp_dir, small_brain_map, small_distance_matrix
    ):
        """Test complete workflow with memory-mapped arrays."""
        # 1. Create distance matrix file
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        # 2. Convert to memmap
        result = txt2memmap(dist_file, temp_dir)

        # 3. Load memmaps
        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        # 4. Create generator
        gen = Sampled(
            small_brain_map, D, index,
            ns=50, knn=50, seed=42
        )

        # 5. Generate surrogates
        surrogates = gen(n=5)

        # 6. Compute statistics
        original = small_brain_map.reshape(1, -1)
        corrs = spearmanr(original, surrogates)

        # Verify
        assert surrogates.shape == (5, 100)
        assert corrs.shape == (1, 5)

    def test_workflow_sorted_arrays(
        self, sorted_distance_matrix, small_brain_map
    ):
        """Test workflow with pre-sorted arrays."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map, D_sorted, index,
            ns=50, knn=50, seed=42
        )

        surrogates = gen(n=3)
        assert surrogates.shape == (3, 100)


class TestStatisticalWorkflow:
    """Integration tests for statistical analysis workflows."""

    def test_null_distribution_workflow(
        self, small_brain_map, small_distance_matrix
    ):
        """Test building a null distribution and computing p-values."""
        # Create two brain maps
        rng = np.random.RandomState(42)
        map1 = small_brain_map
        map2 = rng.randn(100) * 0.3 + map1 * 0.7  # Correlated with map1

        # Generate surrogates for map1
        gen = Base(map1, small_distance_matrix, seed=42)
        surrogates = gen(n=100)

        # Compute observed correlation
        observed_corr = pearsonr(
            map1.reshape(1, -1),
            map2.reshape(1, -1)
        )[0, 0]

        # Build null distribution
        null_corrs = pearsonr(
            map2.reshape(1, -1),
            surrogates
        ).flatten()

        # Compute p-value
        p_value = nonparp(observed_corr, null_corrs)

        # Verify we get a reasonable p-value
        assert 0 <= p_value <= 1

    def test_spearman_workflow(
        self, small_brain_map, small_distance_matrix
    ):
        """Test workflow using Spearman correlation."""
        gen = Base(small_brain_map, small_distance_matrix, seed=42)
        surrogates = gen(n=50)

        # Spearman correlations
        original = small_brain_map.reshape(1, -1)
        corrs = spearmanr(original, surrogates)

        assert corrs.shape == (1, 50)
        # Correlations should be in [-1, 1]
        assert np.all(np.abs(corrs) <= 1)


class TestMaskedDataWorkflow:
    """Integration tests with masked (NaN) data."""

    def test_masked_base_workflow(
        self, small_brain_map_with_nans, small_distance_matrix
    ):
        """Test complete workflow with masked brain map."""
        gen = Base(
            small_brain_map_with_nans, small_distance_matrix,
            seed=42
        )

        surrogates = gen(n=5)
        assert surrogates.shape == (5, 100)

    def test_masked_sampled_workflow(
        self, sorted_distance_matrix, small_brain_map_with_nans
    ):
        """Test Sampled workflow with masked brain map."""
        D_sorted, index = sorted_distance_matrix

        gen = Sampled(
            small_brain_map_with_nans, D_sorted, index,
            ns=50, knn=50, seed=42
        )

        surrogates = gen(n=3)
        assert surrogates.shape == (3, 100)


class TestParameterSweepWorkflow:
    """Integration tests for parameter exploration."""

    def test_delta_sweep(self, small_brain_map, small_distance_matrix):
        """Test workflow exploring different delta values."""
        deltas_options = [
            np.array([0.5]),
            np.array([0.3, 0.6, 0.9]),
            np.linspace(0.1, 0.9, 5),
        ]

        for deltas in deltas_options:
            gen = Base(
                small_brain_map, small_distance_matrix,
                deltas=deltas, seed=42
            )
            surrogate = gen(n=1)
            assert surrogate.shape == (100,)

    def test_nh_sweep(self, small_brain_map, small_distance_matrix):
        """Test workflow with different variogram bin counts."""
        for nh in [10, 25, 50]:
            gen = Base(
                small_brain_map, small_distance_matrix,
                nh=nh, seed=42
            )
            variogram = gen.compute_smooth_variogram(gen.x)
            assert variogram.shape == (nh,)


class TestBatchProcessingWorkflow:
    """Integration tests for batch processing."""

    def test_large_batch_generation(
        self, small_brain_map, small_distance_matrix
    ):
        """Test generating large number of surrogates in batches."""
        gen = Base(
            small_brain_map, small_distance_matrix,
            seed=42, n_jobs=1
        )

        # Generate 50 surrogates with batch size 10
        surrogates = gen(n=50, batch_size=10)

        assert surrogates.shape == (50, 100)

    def test_batch_size_larger_than_n(
        self, small_brain_map, small_distance_matrix
    ):
        """Test when batch_size > n."""
        gen = Base(
            small_brain_map, small_distance_matrix,
            seed=42
        )

        # batch_size > n should still work
        surrogates = gen(n=3, batch_size=100)
        assert surrogates.shape == (3, 100)


class TestErrorHandlingWorkflow:
    """Integration tests for error handling."""

    def test_mismatched_dimensions(self, small_brain_map):
        """Test error when brain map and distance matrix don't match."""
        wrong_size_D = np.random.rand(50, 50)

        with pytest.raises(ValueError):
            Base(small_brain_map, wrong_size_D, seed=42)

    def test_invalid_kernel(self, small_brain_map, small_distance_matrix):
        """Test error with invalid kernel name."""
        with pytest.raises(NotImplementedError):
            Base(
                small_brain_map, small_distance_matrix,
                kernel='invalid_kernel', seed=42
            )

    def test_knn_too_large(self, sorted_distance_matrix, small_brain_map):
        """Test error when knn exceeds nmap."""
        D_sorted, index = sorted_distance_matrix

        with pytest.raises(ValueError):
            Sampled(
                small_brain_map, D_sorted, index,
                ns=50, knn=200,  # knn > nmap (100)
                seed=42
            )
