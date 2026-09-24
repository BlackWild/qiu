"""Unit tests for integer_axis.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_signals.integer_axis import IndexOrdering, IntegerAxis

sizes = st.integers(min_value=1, max_value=64)
orderings = st.sampled_from(list(IndexOrdering))


class TestIndexOrdering:
    """Test the IndexOrdering enum."""

    def test_values(self):
        """Test the list of the raw values."""
        assert IndexOrdering.list() == ["natural", "fft", "centered"]

    @pytest.mark.parametrize("ordering", list(IndexOrdering))
    def test_construction_from_raw_value(self, ordering: IndexOrdering):
        """Test that members are recovered from, and compare equal to, raw values."""
        assert IndexOrdering(ordering.value) is ordering
        assert ordering == ordering.value


class TestIntegerAxis:
    """Test the IntegerAxis."""

    @pytest.mark.parametrize(
        ("ordering", "size", "expected"),
        [
            (IndexOrdering.NATURAL, 4, [0, 1, 2, 3]),
            (IndexOrdering.NATURAL, 5, [0, 1, 2, 3, 4]),
            (IndexOrdering.FFT, 1, [0]),
            (IndexOrdering.FFT, 4, [0, 1, -2, -1]),
            (IndexOrdering.FFT, 5, [0, 1, 2, -2, -1]),
            (IndexOrdering.CENTERED, 1, [0]),
            (IndexOrdering.CENTERED, 4, [-2, -1, 0, 1]),
            (IndexOrdering.CENTERED, 5, [-2, -1, 0, 1, 2]),
        ],
    )
    def test_index_examples(
        self, ordering: IndexOrdering, size: int, expected: list[int]
    ):
        """Test the indices of each ordering on small examples."""
        axis = IntegerAxis(size, ordering)
        np.testing.assert_array_equal(axis.index, expected)
        assert np.issubdtype(axis.index.dtype, np.integer)

    @given(size=sizes)
    def test_index_matches_numpy_fft_conventions(self, size: int):
        """Test the orderings against the conventions of numpy.fft."""
        fft_index = np.rint(np.fft.fftfreq(size) * size).astype(int)

        natural = IntegerAxis(size, IndexOrdering.NATURAL)
        fft = IntegerAxis(size, IndexOrdering.FFT)
        centered = IntegerAxis(size, IndexOrdering.CENTERED)

        np.testing.assert_array_equal(natural.index, np.arange(size))
        np.testing.assert_array_equal(fft.index, fft_index)
        np.testing.assert_array_equal(centered.index, np.fft.fftshift(fft_index))
        np.testing.assert_array_equal(np.fft.ifftshift(centered.index), fft.index)

    @given(size=sizes, ordering=orderings)
    def test_essentials(self, size: int, ordering: IndexOrdering):
        """Test the essential properties of an integer axis."""
        axis = IntegerAxis(size, ordering)

        assert axis.size == size
        assert axis.ordering is ordering
        assert axis.index.shape == (size,)
        assert len(np.unique(axis.index)) == size
        assert repr(axis) == f"IntegerAxis(size={size}, ordering={ordering.value})"

    def test_has_no_physical_attributes(self):
        """Test that an integer axis carries neither a period nor a domain."""
        axis = IntegerAxis(4, IndexOrdering.NATURAL)
        for attribute in ("period", "domain", "values"):
            assert not hasattr(axis, attribute)

    def test_accepts_raw_values(self):
        """Test that the ordering can be given as a raw enum value."""
        axis = IntegerAxis(4, "centered")  # type: ignore
        assert axis.ordering is IndexOrdering.CENTERED

    @pytest.mark.parametrize("size", [0, -1])
    def test_invalid_size(self, size: int):
        """Test that non-positive sizes are rejected."""
        with pytest.raises(ValueError, match="size"):
            IntegerAxis(size, IndexOrdering.NATURAL)

    def test_invalid_ordering(self):
        """Test that unknown orderings are rejected."""
        with pytest.raises(ValueError):
            IntegerAxis(4, "unknown")  # type: ignore


class TestIntegerAxisEquality:
    """Test the equality and hashing of integer axes."""

    def test_equal_axes(self):
        """Test that axes with equal attributes are equal and hash equally."""
        assert IntegerAxis(4, IndexOrdering.FFT) == IntegerAxis(4, "fft")  # type: ignore[arg-type]
        assert hash(IntegerAxis(4, IndexOrdering.FFT)) == hash(
            IntegerAxis(4, IndexOrdering.FFT)
        )
        assert len({IntegerAxis(4, IndexOrdering.FFT), IntegerAxis(4, "fft")}) == 1  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "other",
        [IntegerAxis(5, IndexOrdering.FFT), IntegerAxis(4, IndexOrdering.CENTERED)],
    )
    def test_different_axes(self, other: IntegerAxis):
        """Test that axes differing in any attribute are not equal."""
        assert IntegerAxis(4, IndexOrdering.FFT) != other

    def test_other_objects(self):
        """Test that axes are not equal to unrelated objects."""
        assert IntegerAxis(4, IndexOrdering.FFT) != (4, IndexOrdering.FFT)
