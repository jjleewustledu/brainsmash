"""
Tests for brainsmash.utils.dataio module.

Tests data loading functionality across different file formats.
"""
import numpy as np
import pytest
import os
from brainsmash.utils.dataio import dataio


class TestDataioTextFiles:
    """Tests for loading text files."""

    def test_dataio_txt_whitespace_delimited(self, temp_dir):
        """Test loading whitespace-delimited text file."""
        arr = np.random.rand(100)
        filepath = os.path.join(temp_dir, "data.txt")
        np.savetxt(filepath, arr)

        loaded = dataio(filepath)
        np.testing.assert_array_almost_equal(loaded, arr)

    def test_dataio_txt_space_delimited_2d(self, temp_dir):
        """Test loading space-delimited 2D text file."""
        arr = np.random.rand(10, 10)
        # Note: dataio uses np.loadtxt with default whitespace delimiter
        filepath = os.path.join(temp_dir, "data.txt")
        np.savetxt(filepath, arr, delimiter=' ')

        loaded = dataio(filepath)
        np.testing.assert_array_almost_equal(loaded, arr)

    def test_dataio_txt_2d(self, temp_dir):
        """Test loading 2D text file."""
        arr = np.random.rand(50, 50)
        filepath = os.path.join(temp_dir, "data.txt")
        np.savetxt(filepath, arr)

        loaded = dataio(filepath)
        assert loaded.shape == arr.shape


class TestDataioNpyFiles:
    """Tests for loading numpy binary files."""

    def test_dataio_npy_1d(self, temp_dir):
        """Test loading 1D npy file."""
        arr = np.random.rand(100)
        filepath = os.path.join(temp_dir, "data.npy")
        np.save(filepath, arr)

        loaded = dataio(filepath)
        np.testing.assert_array_equal(loaded, arr)

    def test_dataio_npy_2d(self, temp_dir):
        """Test loading 2D npy file."""
        arr = np.random.rand(50, 50)
        filepath = os.path.join(temp_dir, "data.npy")
        np.save(filepath, arr)

        loaded = dataio(filepath)
        np.testing.assert_array_equal(loaded, arr)

    def test_dataio_npy_preserves_dtype(self, temp_dir):
        """Test that npy loading preserves dtype."""
        arr = np.random.rand(100).astype(np.float32)
        filepath = os.path.join(temp_dir, "data.npy")
        np.save(filepath, arr)

        loaded = dataio(filepath)
        assert loaded.dtype == np.float32


class TestDataioNumpyArrays:
    """Tests for handling numpy arrays directly."""

    def test_dataio_array_passthrough(self):
        """Test that numpy arrays are returned unchanged."""
        arr = np.random.rand(100)
        loaded = dataio(arr)
        np.testing.assert_array_equal(loaded, arr)

    def test_dataio_array_1d(self):
        """Test 1D array passthrough."""
        arr = np.arange(100)
        loaded = dataio(arr)
        assert loaded.shape == (100,)

    def test_dataio_array_2d(self):
        """Test 2D array passthrough."""
        arr = np.random.rand(50, 50)
        loaded = dataio(arr)
        assert loaded.shape == (50, 50)


class TestDataioMemmap:
    """Tests for loading memory-mapped files."""

    def test_dataio_memmap(self, temp_dir):
        """Test loading memory-mapped npy file."""
        arr = np.random.rand(100, 100).astype(np.float32)
        filepath = os.path.join(temp_dir, "data.npy")
        np.save(filepath, arr)

        # Load as memmap
        memmap = np.load(filepath, mmap_mode='r')
        loaded = dataio(memmap)

        np.testing.assert_array_almost_equal(loaded, arr)


class TestDataioEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dataio_empty_array(self):
        """Test handling of empty array."""
        arr = np.array([])
        loaded = dataio(arr)
        assert loaded.size == 0

    def test_dataio_single_element(self):
        """Test handling of single-element array."""
        arr = np.array([42.0])
        loaded = dataio(arr)
        assert loaded.size == 1

    def test_dataio_masked_array(self):
        """Test handling of masked array."""
        arr = np.ma.masked_array(
            data=np.random.rand(100),
            mask=np.zeros(100, dtype=bool)
        )
        loaded = dataio(arr)
        assert loaded.shape == (100,)


class TestDataioIntegration:
    """Integration tests for dataio with other components."""

    def test_dataio_with_base_class(self, temp_dir, small_brain_map, small_distance_matrix):
        """Test that dataio-loaded files work with Base class."""
        from brainsmash.mapgen.base import Base

        # Save files
        map_file = os.path.join(temp_dir, "brain_map.txt")
        dist_file = os.path.join(temp_dir, "distmat.txt")
        np.savetxt(map_file, small_brain_map)
        np.savetxt(dist_file, small_distance_matrix)

        # Load via dataio (implicitly through Base)
        gen = Base(map_file, dist_file, seed=42)

        assert gen.nmap == 100

    def test_dataio_with_npy_files(self, temp_dir, small_brain_map, small_distance_matrix):
        """Test that npy files work with Base class."""
        from brainsmash.mapgen.base import Base

        # Save as npy
        map_file = os.path.join(temp_dir, "brain_map.npy")
        dist_file = os.path.join(temp_dir, "distmat.npy")
        np.save(map_file, small_brain_map)
        np.save(dist_file, small_distance_matrix)

        # Load via Base
        gen = Base(map_file, dist_file, seed=42)

        assert gen.nmap == 100
