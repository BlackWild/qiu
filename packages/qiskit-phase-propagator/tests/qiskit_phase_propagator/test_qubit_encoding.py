"""Unit tests for qubit_encoding.py."""

import numpy as np
import pytest
from python_signals.integer_axis import IndexOrdering, IntegerAxis
from qiskit_phase_propagator.qubit_encoding import (
    bit_weights,
    is_msb_flipped,
    num_qubits_of,
)

all_orderings = pytest.mark.parametrize("ordering", list(IndexOrdering))


class TestNumQubitsOf:
    """Test num_qubits_of."""

    @pytest.mark.parametrize("num_qubits", range(1, 8))
    def test_powers_of_two(self, num_qubits: int):
        """Test that axes of 2**n samples are represented by n qubits."""
        assert (
            num_qubits_of(IntegerAxis(2**num_qubits, IndexOrdering.FFT)) == num_qubits
        )

    @pytest.mark.parametrize("size", [1, 3, 6, 12])
    def test_other_sizes(self, size: int):
        """Test that other sizes are rejected, including a single sample."""
        with pytest.raises(ValueError, match="2\\*\\*n"):
            num_qubits_of(IntegerAxis(size, IndexOrdering.NATURAL))


class TestBitWeights:
    """Test bit_weights and is_msb_flipped."""

    @all_orderings
    @pytest.mark.parametrize("num_qubits", range(1, 7))
    def test_encode_the_axis_indices(self, ordering: IndexOrdering, num_qubits: int):
        """Test that the basis state |k> encodes the index of the sample k."""
        weights = bit_weights(num_qubits, ordering)
        bits = (np.arange(2**num_qubits)[:, None] >> np.arange(num_qubits)) & 1
        if is_msb_flipped(ordering):
            bits[:, -1] ^= 1

        np.testing.assert_array_equal(
            bits @ weights, IntegerAxis(2**num_qubits, ordering).index
        )

    @pytest.mark.parametrize(
        ("ordering", "weights", "flipped"),
        [
            (IndexOrdering.NATURAL, [1, 2, 4], False),
            (IndexOrdering.FFT, [1, 2, -4], False),
            (IndexOrdering.CENTERED, [1, 2, -4], True),
        ],
    )
    def test_examples(self, ordering: IndexOrdering, weights: list[int], flipped: bool):
        """Test the weights of 3 qubits."""
        assert bit_weights(3, ordering) == weights
        assert is_msb_flipped(ordering) is flipped

    def test_accepts_raw_values(self):
        """Test that the ordering can be given as its raw value."""
        assert bit_weights(2, "fft") == [1, -2]  # type: ignore[arg-type]
