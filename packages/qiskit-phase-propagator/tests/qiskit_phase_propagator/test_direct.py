"""Unit tests for direct.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_signals.algebraic_signal import PolynomialSignal
from python_signals.integer_axis import IndexOrdering, IntegerAxis
from python_signals.physical_axis import AxisDomain, PositionAxis
from qiskit_phase_propagator.direct import (
    DIRECT_PHASES,
    DirectPhase,
    polynomial_phase_circuit,
)
from qiskit_pytest_helper.assertions import assert_equal_operators
from qiskit_pytest_helper.hypothesis_strategies import random_polynomial_signal

orderings = st.sampled_from(list(IndexOrdering))
coefs = st.floats(min_value=-1.0, max_value=1.0)
qubits = st.integers(min_value=1, max_value=4)
powers = st.sampled_from(sorted(DIRECT_PHASES))


class TestDirectPhases:
    """Test the direct phase circuits of each power."""

    @given(num_qubits=qubits, coef=coefs, ordering=orderings, power=powers)
    def test_essentials(
        self, num_qubits: int, coef: float, ordering: IndexOrdering, power: int
    ):
        """Test the attributes of the phase circuits."""
        circuit = DIRECT_PHASES[power](num_qubits, coef, ordering)

        assert isinstance(circuit, DirectPhase)
        assert circuit.exponent == power
        assert circuit.num_qubits == num_qubits
        assert circuit.coef == coef
        assert circuit.ordering is ordering

    @given(num_qubits=qubits, coef=coefs, ordering=orderings, power=powers)
    def test_applies_the_phase(
        self, num_qubits: int, coef: float, ordering: IndexOrdering, power: int
    ):
        """Test that the basis state |k> gets the phase e^(i coef index[k]^power)."""
        circuit = DIRECT_PHASES[power](num_qubits, coef, ordering)
        index = IntegerAxis(2**num_qubits, ordering).index

        assert_equal_operators(circuit, np.diag(np.exp(1j * coef * index**power)))

    def test_accepts_raw_orderings(self):
        """Test that the ordering can be given as its raw value."""
        circuit = DIRECT_PHASES[1](2, 0.5, "centered")  # type: ignore[arg-type]
        assert circuit.ordering is IndexOrdering.CENTERED


class TestPolynomialPhaseCircuit:
    """Test polynomial_phase_circuit."""

    @given(
        signal=st.integers(min_value=1, max_value=3).flatmap(
            lambda degree: random_polynomial_signal(
                degree=degree, domain=AxisDomain.POSITION
            )
        )
    )
    def test_applies_the_signal(self, signal: PolynomialSignal):
        """Test that the basis state |k> gets the phase e^(i signal(x_k))."""
        circuit = polynomial_phase_circuit(signal)
        assert_equal_operators(circuit, np.diag(np.exp(1j * signal.data)))

    def test_constant_signal(self):
        """Test that a constant signal is a global phase."""
        signal = PolynomialSignal(PositionAxis(4, 1.0, IndexOrdering.FFT), 0.3, 0)
        circuit = polynomial_phase_circuit(signal)
        assert circuit.size() == 0
        assert_equal_operators(circuit, np.exp(0.3j) * np.eye(4))

    def test_higher_powers_are_not_implemented(self):
        """Test that powers above 3 are rejected."""
        signal = PolynomialSignal(PositionAxis(4, 1.0, IndexOrdering.FFT), 0.3, 4)
        with pytest.raises(NotImplementedError, match="up to 3"):
            polynomial_phase_circuit(signal)

    def test_axes_must_have_qubit_sizes(self):
        """Test that the axis must have 2**n samples."""
        signal = PolynomialSignal(PositionAxis(6, 1.0, IndexOrdering.FFT), 0.3, 2)
        with pytest.raises(ValueError, match="2\\*\\*n"):
            polynomial_phase_circuit(signal)
