"""
Tests for brainsmash.mapgen.memmap module.

Tests memory-mapped array creation and loading functionality.
"""
import numpy as np
import pytest
import os
from brainsmash.mapgen.memmap import txt2memmap, load_memmap


class TestTxt2Memmap:
    """Tests for txt2memmap function."""

    def test_txt2memmap_basic(self, temp_dir, small_distance_matrix):
        """Test basic txt2memmap functionality."""
        # Write distance matrix to text file
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        # Convert to memmap
        result = txt2memmap(dist_file, temp_dir)

        assert 'distmat' in result
        assert 'index' in result
        assert os.path.exists(result['distmat'])
        assert os.path.exists(result['index'])

    def test_txt2memmap_creates_sorted_arrays(
        self, temp_dir, small_distance_matrix
    ):
        """Verify that rows are sorted in the output."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        result = txt2memmap(dist_file, temp_dir)

        # Load and verify sorting
        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        # Each row should be sorted in ascending order
        for i in range(D.shape[0]):
            assert np.all(np.diff(D[i]) >= 0), f"Row {i} is not sorted"

    def test_txt2memmap_index_correctness(
        self, temp_dir, small_distance_matrix
    ):
        """Verify that index array correctly maps to sorted distances."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        result = txt2memmap(dist_file, temp_dir)

        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        # Verify sorting: each row of D should be sorted
        for i in range(min(5, D.shape[0])):
            assert np.all(np.diff(D[i]) >= 0), f"Row {i} is not sorted"

        # Verify index shape matches D shape
        assert D.shape == index.shape

    def test_txt2memmap_output_dtype(self, temp_dir, small_distance_matrix):
        """Verify output dtypes."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        result = txt2memmap(dist_file, temp_dir)

        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        assert D.dtype == np.float32
        assert index.dtype == np.int32

    def test_txt2memmap_nonexistent_output_dir(self, temp_dir):
        """txt2memmap should raise IOError for non-existent output directory."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, np.eye(10), delimiter=' ')

        with pytest.raises(IOError):
            txt2memmap(dist_file, "/nonexistent/directory")

    def test_txt2memmap_with_mask(self, temp_dir, small_distance_matrix):
        """Test txt2memmap with a mask file."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        # Create mask (mask first 10 elements)
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True
        mask_file = os.path.join(temp_dir, "mask.txt")
        np.savetxt(mask_file, mask.astype(int), fmt='%i')

        result = txt2memmap(dist_file, temp_dir, maskfile=mask_file)

        D = load_memmap(result['distmat'])

        # Output should have 90 rows (100 - 10 masked)
        assert D.shape[0] == 90
        assert D.shape[1] == 90

    def test_txt2memmap_mask_size_mismatch(self, temp_dir, small_distance_matrix):
        """txt2memmap should raise ValueError for mask size mismatch."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        # Create mask with wrong size
        mask = np.zeros(50, dtype=bool)  # Wrong size
        mask_file = os.path.join(temp_dir, "mask.txt")
        np.savetxt(mask_file, mask.astype(int), fmt='%i')

        with pytest.raises(ValueError):
            txt2memmap(dist_file, temp_dir, maskfile=mask_file)

    def test_txt2memmap_custom_delimiter(self, temp_dir, small_distance_matrix):
        """Test txt2memmap with custom delimiter."""
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=',')

        result = txt2memmap(dist_file, temp_dir, delimiter=',')

        D = load_memmap(result['distmat'])
        assert D.shape == (100, 100)


class TestLoadMemmap:
    """Tests for load_memmap function."""

    def test_load_memmap_basic(self, temp_dir):
        """Test basic load_memmap functionality."""
        # Create and save array
        arr = np.random.rand(50, 50).astype(np.float32)
        filepath = os.path.join(temp_dir, "test.npy")
        np.save(filepath, arr)

        # Load as memmap
        loaded = load_memmap(filepath)

        np.testing.assert_array_almost_equal(loaded, arr)

    def test_load_memmap_is_memmap(self, temp_dir):
        """Verify that load_memmap returns a memory-mapped array."""
        arr = np.random.rand(50, 50).astype(np.float32)
        filepath = os.path.join(temp_dir, "test.npy")
        np.save(filepath, arr)

        loaded = load_memmap(filepath)

        assert isinstance(loaded, np.memmap)

    def test_load_memmap_read_only(self, temp_dir):
        """Verify that loaded memmap is read-only."""
        arr = np.random.rand(50, 50).astype(np.float32)
        filepath = os.path.join(temp_dir, "test.npy")
        np.save(filepath, arr)

        loaded = load_memmap(filepath)

        # Should be read-only
        with pytest.raises((ValueError, TypeError)):
            loaded[0, 0] = 999.0


class TestMemmapRoundTrip:
    """Integration tests for memmap creation and loading."""

    def test_full_roundtrip(self, temp_dir, small_distance_matrix):
        """Test complete txt2memmap -> load_memmap workflow."""
        # Save original matrix
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')

        # Convert to memmap
        result = txt2memmap(dist_file, temp_dir)

        # Load memmaps
        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        # Verify we can reconstruct original matrix
        reconstructed = np.zeros_like(small_distance_matrix)
        for i in range(D.shape[0]):
            reconstructed[i, index[i]] = D[i]

        np.testing.assert_array_almost_equal(
            reconstructed, small_distance_matrix, decimal=5
        )

    def test_memmap_usable_with_sampled(
        self, temp_dir, small_distance_matrix, small_brain_map
    ):
        """Test that memmap files work with Sampled class."""
        from brainsmash.mapgen.sampled import Sampled

        # Create memmaps
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(dist_file, small_distance_matrix, delimiter=' ')
        result = txt2memmap(dist_file, temp_dir)

        # Load memmaps
        D = load_memmap(result['distmat'])
        index = load_memmap(result['index'])

        # Should work with Sampled class
        gen = Sampled(
            small_brain_map,
            D,
            index,
            ns=50,
            knn=50,
            seed=42
        )

        surrogate = gen(n=1)
        assert surrogate.shape == (100,)
