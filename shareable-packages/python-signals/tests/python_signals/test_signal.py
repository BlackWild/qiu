"""Unit tests for signal.py."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import (
    position_axes,
    sampled_signals,
)
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PhysicalAxis, PositionAxis
from python_signals.signal import Signal


class TestSignal:
    """Test the Signal."""

    @given(axis=position_axes())
    def test_essentials(self, axis: PhysicalAxis):
        """Test the essential properties of a signal."""
        data = np.cos(axis.values)
        signal = Signal(axis, data)

        assert signal.axis is axis
        assert signal.size == axis.size
        np.testing.assert_array_equal(signal.data, data)
        assert type(signal).__name__ in repr(signal)

    def test_accepts_array_likes(self):
        """Test that the data can be given as any array-like."""
        signal = Signal(PositionAxis(3, 1.0, IndexOrdering.NATURAL), [1, 2, 3])
        assert isinstance(signal.data, np.ndarray)
        np.testing.assert_array_equal(signal.data, [1, 2, 3])

    def test_complex_data(self):
        """Test that complex sampled values are supported."""
        axis = PositionAxis(8, 0.5, IndexOrdering.FFT)
        signal = Signal(axis, np.exp(1j * axis.values))
        assert np.iscomplexobj(signal.data)

    @pytest.mark.parametrize("shape", [(3,), (5,), (4, 1), ()])
    def test_invalid_shape(self, shape: tuple[int, ...]):
        """Test that the data must have one value per axis sample."""
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(ValueError, match="shape"):
            Signal(axis, np.zeros(shape))

    def test_invalid_dtype(self):
        """Test that the data must be numeric."""
        axis = PositionAxis(2, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(ValueError, match="numeric"):
            Signal(axis, ["a", "b"])

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @given(data=st.data())
    def test_normalized_data(self, dtype: npt.DTypeLike, data: st.DataObject):
        """Test that the normalized data has unit norm and the same direction."""
        signal = data.draw(sampled_signals(dtype=dtype))
        normalized = signal.normalized_data

        if not np.any(signal.data):
            np.testing.assert_array_equal(normalized, signal.data)
            return
        assert_close(np.linalg.norm(normalized), 1.0)
        # the data is its norm, the projection onto the normalized data, times it;
        # the projection is real and positive, i.e. equal to its magnitude
        projection = np.vdot(normalized, signal.data)
        norm = float(abs(projection))
        assert_close(projection, norm)
        # the normalized samples may underflow before being scaled by the norm
        assert_close(normalized * norm, signal.data, scale=norm)

    @pytest.mark.parametrize("magnitude", [1e-200, 1e-160, 1e160, 1e300])
    def test_normalized_data_of_extreme_magnitudes(self, magnitude: float):
        """Test that tiny and huge signals are normalized without under- or overflow."""
        signal = Signal(PositionAxis(3, 1.0, IndexOrdering.NATURAL), [3.0, 0.0, 4.0])
        scaled = Signal(signal.axis, magnitude * signal.data)
        assert_close(scaled.normalized_data, [0.6, 0.0, 0.8])

    def test_normalized_data_of_zero_signal(self):
        """Test that an all-zero signal is returned unchanged."""
        signal = Signal(PositionAxis(4, 1.0, IndexOrdering.NATURAL), np.zeros(4))
        np.testing.assert_array_equal(signal.normalized_data, np.zeros(4))
