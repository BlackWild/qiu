"""Unit tests for signal.py."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PhysicalAxis, PositionAxis
from python_signals.signal import Signal


@st.composite
def position_axes(draw) -> PositionAxis:
    """A strategy for generating position axes."""
    return PositionAxis(
        size=draw(st.integers(min_value=1, max_value=64)),
        delta_x=draw(st.floats(min_value=1e-3, max_value=10.0)),
        ordering=draw(st.sampled_from(list(IndexOrdering))),
    )


@st.composite
def signals(draw, dtype: npt.DTypeLike = np.float64) -> Signal:
    """A strategy for generating signals with finite sampled values."""
    axis = draw(position_axes())
    elements = (
        st.complex_numbers(max_magnitude=1e3, allow_nan=False, allow_infinity=False)
        if np.issubdtype(dtype, np.complexfloating)
        else st.floats(min_value=-1e3, max_value=1e3)
    )
    return Signal(axis, draw(arrays(dtype, axis.size, elements=elements)))


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

    @given(signal=signals())
    def test_normalized_data(self, signal: Signal):
        """Test that the normalized data has unit norm and the same direction."""
        norm = np.linalg.norm(signal.data)
        normalized = signal.normalized_data

        if norm == 0:
            np.testing.assert_array_equal(normalized, signal.data)
        else:
            assert np.isclose(np.linalg.norm(normalized), 1.0)
            np.testing.assert_allclose(normalized * norm, signal.data, atol=1e-9)

    @given(signal=signals(dtype=np.complex128))
    def test_normalized_complex_data(self, signal: Signal):
        """Test that complex signals are normalized by their Euclidean norm."""
        if np.linalg.norm(signal.data) > 0:
            assert np.isclose(np.linalg.norm(signal.normalized_data), 1.0)

    def test_normalized_data_of_zero_signal(self):
        """Test that an all-zero signal is returned unchanged."""
        signal = Signal(PositionAxis(4, 1.0, IndexOrdering.NATURAL), np.zeros(4))
        np.testing.assert_array_equal(signal.normalized_data, np.zeros(4))
