"""Unit tests for assertions.py."""

import numpy as np
import pytest
from python_pytest_helper.assertions import (
    RTOL,
    assert_close,
    is_close,
    underflow_atol,
)

TINY = np.finfo(np.float64).tiny
SMALLEST_SUBNORMAL = np.nextafter(0.0, 1.0)


class TestAssertClose:
    """Test assert_close and is_close."""

    def test_accepts_rounding_errors(self):
        """Test that values agreeing up to the relative tolerance are close."""
        assert_close(1.0 + RTOL / 2, 1.0)
        assert_close([1e300, -1e-300], [1e300 * (1 + RTOL / 2), -1e-300])
        assert is_close(1.0 + RTOL / 2, 1.0)

    def test_rejects_larger_errors(self):
        """Test that values differing by more than the relative tolerance fail."""
        with pytest.raises(AssertionError):
            assert_close(1.0 + 10 * RTOL, 1.0)
        assert not is_close(1.0 + 10 * RTOL, 1.0)

    def test_relative_at_every_normal_scale(self):
        """Test that small normal values are not compared absolutely."""
        assert not is_close(1e-300, 2e-300)
        assert not is_close(1e-300, 0.0)

    def test_accepts_underflow(self):
        """Test that underflowed values only need to agree absolutely."""
        assert_close(0.0, SMALLEST_SUBNORMAL)
        assert_close([2.0, 0.0], [2.0, SMALLEST_SUBNORMAL])

    def test_scaled_underflow(self):
        """Test that the underflow tolerance scales with the given factor."""
        assert not is_close(0.0, 2 * TINY)
        assert_close(0.0, 2 * TINY, scale=2.0)

    def test_complex_values(self):
        """Test that complex values are compared by their absolute difference."""
        assert_close(1j * (1 + RTOL / 2), 1j)
        assert not is_close(1j, -1j)

    def test_exact_types(self):
        """Test that integers are compared as floats."""
        assert_close(np.arange(3), [0.0, 1.0, 2.0])


class TestUnderflowAtol:
    """Test underflow_atol."""

    def test_smallest_normal_float(self):
        """Test that the tolerance is the smallest normal float of the type."""
        assert underflow_atol() == TINY
        assert underflow_atol(np.complex128) == TINY
        assert underflow_atol(np.float32) == np.finfo(np.float32).tiny

    def test_scale(self):
        """Test that the tolerance scales with the magnitude of larger factors."""
        assert underflow_atol(scale=-3.0) == 3 * TINY

    def test_scaling_down_keeps_the_underflow_of_the_result(self):
        """Test that smaller factors keep the tolerance of the result's underflow."""
        assert underflow_atol(scale=1e-300) == TINY
        assert_close(2 * SMALLEST_SUBNORMAL, SMALLEST_SUBNORMAL, scale=1e-300)
