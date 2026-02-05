"""
Tests for numpy API compatibility with Python 3.13.

These tests specifically verify that no deprecated numpy patterns
cause issues when running on Python 3.13 with numpy 1.26+.
"""
import numpy as np
import pytest
import sys


class TestNumpyDtypes:
    """Tests for numpy dtype compatibility."""

    def test_int32_dtype(self):
        """Verify np.int32 is available and working."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        assert arr.dtype == np.int32

    def test_int64_dtype(self):
        """Verify np.int64 is available and working."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        assert arr.dtype == np.int64

    def test_float32_dtype(self):
        """Verify np.float32 is available and working."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert arr.dtype == np.float32

    def test_float64_dtype(self):
        """Verify np.float64 is available and working."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert arr.dtype == np.float64

    def test_dtype_from_string(self):
        """Verify dtype creation from string."""
        arr32 = np.array([1, 2, 3], dtype='int32')
        arr64 = np.array([1.0, 2.0], dtype='float64')

        assert arr32.dtype == np.int32
        assert arr64.dtype == np.float64

    def test_iinfo_int32(self):
        """Verify np.iinfo works with np.int32."""
        info = np.iinfo(np.int32)
        assert info.max == 2147483647
        assert info.min == -2147483648

    def test_finfo_float64(self):
        """Verify np.finfo works with np.float64."""
        info = np.finfo(np.float64)
        assert info.max > 1e300
        assert info.eps < 1e-15


class TestRandomStateCompat:
    """Tests for numpy RandomState compatibility."""

    def test_random_method(self):
        """Verify RandomState.random() works (replacement for random_sample)."""
        rs = np.random.RandomState(42)
        arr = rs.random((10, 5))

        assert arr.shape == (10, 5)
        assert np.all(arr >= 0)
        assert np.all(arr < 1)

    def test_randint(self):
        """Verify RandomState.randint() works."""
        rs = np.random.RandomState(42)
        val = rs.randint(np.iinfo(np.int32).max)

        assert isinstance(val, (int, np.integer))
        assert 0 <= val < np.iinfo(np.int32).max

    def test_randn(self):
        """Verify RandomState.randn() works."""
        rs = np.random.RandomState(42)
        arr = rs.randn(100)

        assert arr.shape == (100,)
        # Should be approximately standard normal
        assert -5 < arr.mean() < 5
        assert 0.5 < arr.std() < 2

    def test_permutation(self):
        """Verify RandomState.permutation() works."""
        rs = np.random.RandomState(42)
        perm = rs.permutation(100)

        assert perm.shape == (100,)
        assert len(np.unique(perm)) == 100

    def test_choice(self):
        """Verify RandomState.choice() works."""
        rs = np.random.RandomState(42)
        choice = rs.choice(100, size=50, replace=False)

        assert choice.shape == (50,)
        assert len(np.unique(choice)) == 50


class TestMemmapCompat:
    """Tests for memory-mapped array compatibility."""

    def test_memmap_creation(self, temp_dir):
        """Verify memory-mapped array creation."""
        import os
        import numpy.lib.format

        filepath = os.path.join(temp_dir, "test.npy")
        shape = (100, 100)

        # Create memmap using numpy.lib.format
        fp = numpy.lib.format.open_memmap(
            filepath, mode='w+', dtype=np.float32, shape=shape
        )
        fp[:] = np.random.rand(*shape).astype(np.float32)
        del fp  # Flush to disk

        # Verify file exists and can be loaded
        loaded = np.load(filepath, mmap_mode='r')
        assert loaded.shape == shape

    def test_memmap_int32_index(self, temp_dir):
        """Verify int32 index array creation."""
        import os
        import numpy.lib.format

        filepath = os.path.join(temp_dir, "index.npy")
        shape = (100, 100)

        fp = numpy.lib.format.open_memmap(
            filepath, mode='w+', dtype=np.int32, shape=shape
        )
        fp[:] = np.arange(10000).reshape(shape).astype(np.int32)
        del fp

        loaded = np.load(filepath, mmap_mode='r')
        assert loaded.dtype == np.int32


class TestArrayOperationsCompat:
    """Tests for array operations compatibility."""

    def test_argsort(self):
        """Verify argsort works as expected."""
        arr = np.random.rand(100)
        idx = np.argsort(arr)

        # Verify sorting
        assert np.all(np.diff(arr[idx]) >= 0)

    def test_take_along_axis(self):
        """Verify take_along_axis works."""
        arr = np.random.rand(10, 5)
        indices = np.argmax(arr, axis=1, keepdims=True)
        result = np.take_along_axis(arr, indices, axis=1)

        assert result.shape == (10, 1)

    def test_put_along_axis(self):
        """Verify put_along_axis works."""
        arr = np.zeros((10, 5))
        indices = np.array([[0], [1], [2], [3], [4], [0], [1], [2], [3], [4]])
        values = np.ones((10, 1))

        np.put_along_axis(arr, indices, values, axis=1)
        assert np.sum(arr) == 10

    def test_triu_indices(self):
        """Verify triu_indices works."""
        idx = np.triu_indices(100, k=1)
        assert len(idx) == 2
        assert len(idx[0]) == 100 * 99 // 2

    def test_masked_array(self):
        """Verify masked array operations work."""
        data = np.random.rand(100)
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True

        ma = np.ma.masked_array(data=data, mask=mask)
        assert ma.count() == 90
        assert np.sum(ma.mask) == 10


class TestLinearAlgebraCompat:
    """Tests for linear algebra compatibility."""

    def test_dot_product(self):
        """Verify matrix dot product works."""
        A = np.random.rand(10, 5)
        B = np.random.rand(5, 10)
        result = np.dot(A, B)

        assert result.shape == (10, 10)

    def test_matrix_operations(self):
        """Verify matrix operations work."""
        A = np.random.rand(10, 10)

        # Various operations
        assert A.T.shape == (10, 10)
        assert np.sum(A, axis=0).shape == (10,)
        assert np.mean(A, axis=1).shape == (10,)


class TestBrainsmashSpecificCompat:
    """Tests specific to brainsmash code patterns."""

    def test_base_permute_pattern(self):
        """Test the pattern used in Base.permute_map()."""
        rs = np.random.RandomState(42)
        x_size = 100
        i = 5

        # This is the updated pattern (using .random() instead of .random_sample())
        perm_idx = rs.random((x_size, i)).argsort(axis=0)

        assert perm_idx.shape == (100, 5)
        # Each column should be a valid permutation
        for col in range(i):
            assert len(np.unique(perm_idx[:, col])) == x_size

    def test_randint_with_iinfo(self):
        """Test randint with np.iinfo pattern used in __call__."""
        rs = np.random.RandomState(42)
        size = 10

        rs_values = rs.randint(np.iinfo(np.int32).max, size=size)

        assert rs_values.shape == (size,)
        assert np.all(rs_values >= 0)
        assert np.all(rs_values < np.iinfo(np.int32).max)

    def test_astype_int32(self):
        """Test astype to int32 pattern used in Sampled.sample()."""
        rs = np.random.RandomState(42)
        nmap = 100
        ns = 50

        result = rs.choice(a=nmap, size=ns, replace=False).astype(np.int32)

        assert result.dtype == np.int32
        assert result.shape == (ns,)


class TestPythonVersionInfo:
    """Tests to verify Python version information."""

    def test_python_version(self):
        """Log Python version for debugging."""
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        assert version.major == 3
        assert version.minor >= 10  # Minimum supported version

    def test_numpy_version(self):
        """Log numpy version for debugging."""
        print(f"NumPy version: {np.__version__}")
        # Verify minimum version for Python 3.13 support
        major, minor = map(int, np.__version__.split('.')[:2])
        assert major >= 1
        if major == 1:
            assert minor >= 26  # numpy 1.26+ required for Python 3.13
