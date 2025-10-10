"""Unit tests for direct.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_phase_propagator.direct import (
    Order1DirectPhase,
    Order2DirectPhase,
    Order3DirectPhase,
)


class TestOrder1DirectPhase:
    """Test the Order1DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the essentials of the Order1DirectPhase."""

        propagator = Order1DirectPhase(num_qubits, coef, signed)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.signed == signed

    @given(
        num_qubits=st.integers(min_value=1, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the correct phase application of the Order1DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order1DirectPhase(num_qubits, coef, signed)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x = (
            np.concatenate(
                [
                    np.arange(0, 2 ** (num_qubits - 1)),
                    np.arange(-(2 ** (num_qubits - 1)), 0),
                ]
            )
            if signed
            else np.arange(2**num_qubits)
        )
        expected_output_state = np.exp(1j * coef * x) * psi.data

        assert np.allclose(output_state.data, expected_output_state)


class TestOrder2DirectPhase:
    """Test the Order2DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the essentials of the Order2DirectPhase."""

        propagator = Order2DirectPhase(num_qubits, coef, signed)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.signed == signed

    @given(
        num_qubits=st.integers(min_value=1, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the correct phase application of the Order2DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order2DirectPhase(num_qubits, coef, signed)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x = (
            np.concatenate(
                [
                    np.arange(0, 2 ** (num_qubits - 1)),
                    np.arange(-(2 ** (num_qubits - 1)), 0),
                ]
            )
            if signed
            else np.arange(2**num_qubits)
        )
        expected_output_state = np.exp(1j * coef * x**2) * psi.data

        assert np.allclose(output_state.data, expected_output_state)


class TestOrder3DirectPhase:
    """Test the Order3DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the essentials of the Order3DirectPhase."""

        propagator = Order3DirectPhase(num_qubits, coef, signed)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.signed == signed

    @given(
        num_qubits=st.integers(min_value=1, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        signed=st.booleans(),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        signed: bool,
    ):
        """Test the correct phase application of the Order3DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order3DirectPhase(num_qubits, coef, signed)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x = (
            np.concatenate(
                [
                    np.arange(0, 2 ** (num_qubits - 1)),
                    np.arange(-(2 ** (num_qubits - 1)), 0),
                ]
            )
            if signed
            else np.arange(2**num_qubits)
        )
        expected_output_state = np.exp(1j * coef * x**3) * psi.data

        assert np.allclose(output_state.data, expected_output_state)
