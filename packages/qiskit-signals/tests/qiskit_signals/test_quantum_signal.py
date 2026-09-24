"""Unit tests for quantum_signal.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_signals.algebraic_signal import (
    AlgebraicSignal,
    PolynomialSignal,
    QuadraticSignal,
)
from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import GenericAxis, PositionAxis
from qiskit_signals.quantum_signal import (
    GenericQuantumSignal,
    PolynomialQuantumSignal,
    QuadraticQuantumSignal,
)


@st.composite
def position_axes(draw) -> PositionAxis:
    """A strategy for generating quantum position axes."""
    return PositionAxis(
        num_qubits=draw(st.integers(min_value=1, max_value=6)),
        delta_x=draw(st.floats(min_value=1e-3, max_value=10.0)),
        encoding=draw(st.sampled_from(list(EncodingType))),
    )


alphas = st.floats(min_value=-10.0, max_value=10.0)


class TestGenericQuantumSignal:
    """Test the GenericQuantumSignal."""

    @given(axis=position_axes())
    def test_essentials(self, axis: GenericAxis):
        """Test the essential properties of a generic quantum signal."""
        signal = GenericQuantumSignal(axis, np.cos)

        assert isinstance(signal, AlgebraicSignal)
        assert signal.signal_function is np.cos
        assert np.isclose(np.linalg.norm(signal.normalized_data), 1.0)
        assert signal.num_qubits == axis.num_qubits
        assert signal.dimension == signal.size == 2**axis.num_qubits
        np.testing.assert_array_equal(signal.data, np.cos(axis.axis_values))


class TestPolynomialQuantumSignal:
    """Test the PolynomialQuantumSignal."""

    @given(axis=position_axes(), alpha=alphas, power=st.integers(0, 3))
    def test_essentials(self, axis: GenericAxis, alpha: float, power: int):
        """Test the essential properties of a polynomial quantum signal."""
        signal = PolynomialQuantumSignal(axis, alpha, power)

        assert isinstance(signal, PolynomialSignal)
        assert signal.num_qubits == axis.num_qubits
        assert signal.encoding is axis.encoding
        np.testing.assert_allclose(signal.data, alpha * axis.axis_values**power)

    @given(axis=position_axes(), alpha=alphas, power=st.integers(0, 3))
    def test_to_generic(self, axis: GenericAxis, alpha: float, power: int):
        """Test the conversion to a generic quantum signal."""
        polynomial = PolynomialQuantumSignal(axis, alpha, power)
        generic = polynomial.to_generic()

        assert isinstance(generic, GenericQuantumSignal)
        assert generic.axis is axis
        np.testing.assert_allclose(generic.data, polynomial.data)


class TestQuadraticQuantumSignal:
    """Test the QuadraticQuantumSignal."""

    @given(axis=position_axes(), alpha=alphas)
    def test_essentials(self, axis: GenericAxis, alpha: float):
        """Test that a quadratic quantum signal is both quantum and quadratic."""
        signal = QuadraticQuantumSignal(axis=axis, alpha=alpha)

        assert isinstance(signal, PolynomialQuantumSignal)
        assert isinstance(signal, QuadraticSignal)
        assert signal.power == 2
        assert signal.encoding is axis.encoding
        np.testing.assert_allclose(signal.effective_alpha, alpha * axis.period**2)
