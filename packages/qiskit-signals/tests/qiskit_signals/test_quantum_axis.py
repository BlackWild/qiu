"""Unit tests for quantum_axis.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain, PhysicalAxis
from qiskit_signals.helper_types import AxisType, EncodingType
from qiskit_signals.quantum_axis import (
    AngularWavenumberAxis,
    MomentumAxis,
    PositionAxis,
    SpatialFrequencyAxis,
)

num_qubits = st.integers(min_value=1, max_value=6)
delta_xs = st.floats(min_value=1e-3, max_value=10.0)
encodings = st.sampled_from(list(EncodingType))


def expected_index(num_qubits: int, encoding: EncodingType) -> np.ndarray:
    """The integer represented by each computational basis state."""
    half = 2 ** (num_qubits - 1)
    if encoding == EncodingType.UNSIGNED:
        return np.arange(2 * half)
    if encoding == EncodingType.TWOS_COMPLEMENT:
        return np.concatenate((np.arange(0, half), np.arange(-half, 0)))
    return np.concatenate((np.arange(-half, 0), np.arange(0, half)))


class TestEncodingType:
    """Test the EncodingType enum."""

    @pytest.mark.parametrize(
        ("encoding", "ordering"),
        [
            (EncodingType.UNSIGNED, IndexOrdering.NATURAL),
            (EncodingType.TWOS_COMPLEMENT, IndexOrdering.FFT),
            (EncodingType.TWOS_COMPLEMENT_MIRRORED, IndexOrdering.CENTERED),
        ],
    )
    def test_index_ordering(self, encoding: EncodingType, ordering: IndexOrdering):
        """Test the index ordering corresponding to each encoding."""
        assert encoding.index_ordering is ordering

    def test_axis_type_is_axis_domain(self):
        """Test that the axis types are the domains of python_signals."""
        assert AxisType is AxisDomain


class TestPositionAxis:
    """Test the quantum PositionAxis."""

    @given(num_qubits=num_qubits, delta_x=delta_xs, encoding=encodings)
    def test_essentials(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Test the essential properties of a quantum position axis."""
        axis = PositionAxis(num_qubits, delta_x, encoding)

        assert isinstance(axis, PhysicalAxis)
        assert axis.num_qubits == num_qubits
        assert axis.encoding is encoding
        assert axis.dimension == axis.size == 2**num_qubits
        assert axis.axis_type is AxisType.POSITION
        assert not axis.is_fourier_domain_axis
        np.testing.assert_array_equal(axis.index, expected_index(num_qubits, encoding))
        np.testing.assert_array_equal(axis.axis_values, axis.index * delta_x)

    @pytest.mark.parametrize("encoding", EncodingType.list())
    def test_accepts_raw_values(self, encoding: str):
        """Test that the encoding can be given as its raw value."""
        axis = PositionAxis(3, 1.0, encoding)  # type: ignore
        assert axis.encoding is EncodingType(encoding)


class TestFourierConjugateAxes:
    """Test the quantum axes conjugate to a position axis."""

    @given(num_qubits=num_qubits, delta_x=delta_xs, encoding=encodings)
    def test_periods(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Test the periods and domains of the conjugate axes."""
        window = 2**num_qubits * delta_x
        axes = [
            (MomentumAxis(num_qubits, delta_x, encoding, hbar=2.0), 4 * np.pi / window),
            (AngularWavenumberAxis(num_qubits, delta_x, encoding), 2 * np.pi / window),
            (SpatialFrequencyAxis(num_qubits, delta_x, encoding), 1 / window),
        ]
        for axis, period in axes:
            assert np.isclose(axis.period, period)
            assert axis.is_fourier_domain_axis
            assert axis.encoding is encoding
            np.testing.assert_array_equal(
                axis.index, expected_index(num_qubits, encoding)
            )

    @given(num_qubits=num_qubits, delta_x=delta_xs, encoding=encodings)
    def test_from_position_axis(
        self, num_qubits: int, delta_x: float, encoding: EncodingType
    ):
        """Test that the conjugate axes default to the two's complement encoding."""
        x_axis = PositionAxis(num_qubits, delta_x, encoding)

        p_axis = MomentumAxis.from_position_axis(x_axis, hbar=1.0)
        k_axis = AngularWavenumberAxis.from_position_axis(x_axis)
        assert p_axis.encoding is EncodingType.TWOS_COMPLEMENT
        assert k_axis.encoding is EncodingType.TWOS_COMPLEMENT
        np.testing.assert_allclose(k_axis.axis_values, p_axis.axis_values)

        kept = MomentumAxis.from_position_axis(x_axis, hbar=1.0, keep_encoding=True)
        assert kept.encoding is encoding
