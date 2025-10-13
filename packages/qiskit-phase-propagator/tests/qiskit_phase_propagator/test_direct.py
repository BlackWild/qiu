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
from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import PositionAxis


class TestOrder1DirectPhase:
    """Test the Order1DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the essentials of the Order1DirectPhase."""

        propagator = Order1DirectPhase(num_qubits, coef, encoding)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.encoding == encoding

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the correct phase application of the Order1DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order1DirectPhase(num_qubits, coef, encoding)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x_axis = PositionAxis(delta_x=1.0, num_qubits=num_qubits, encoding=encoding)
        x = x_axis.axis_values
        expected_output_state = np.exp(1j * coef * x) * psi.data

        assert np.allclose(output_state.data, expected_output_state), (
            f"{output_state.data} vs {expected_output_state} with x={x}"
        )


class TestOrder2DirectPhase:
    """Test the Order2DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the essentials of the Order2DirectPhase."""

        propagator = Order2DirectPhase(num_qubits, coef, encoding)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.encoding == encoding

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the correct phase application of the Order2DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order2DirectPhase(num_qubits, coef, encoding)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x_axis = PositionAxis(delta_x=1.0, num_qubits=num_qubits, encoding=encoding)
        x = x_axis.axis_values
        expected_output_state = np.exp(1j * coef * x**2) * psi.data

        assert np.allclose(output_state.data, expected_output_state)


class TestOrder3DirectPhase:
    """Test the Order3DirectPhase."""

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_essentials(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the essentials of the Order3DirectPhase."""

        propagator = Order3DirectPhase(num_qubits, coef, encoding)
        assert propagator.num_qubits == num_qubits
        assert propagator.coef == coef
        assert propagator.encoding == encoding

    @given(
        num_qubits=st.integers(min_value=2, max_value=4),
        coef=st.floats(min_value=-1.0, max_value=1.0),
        encoding=st.sampled_from(EncodingType.list()),
    )
    def test_correct_phase_application(
        self,
        num_qubits: int,
        coef: float,
        encoding: EncodingType,
    ):
        """Test the correct phase application of the Order3DirectPhase."""

        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = Order3DirectPhase(num_qubits, coef, encoding)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output_state = Statevector(circuit)
        x_axis = PositionAxis(delta_x=1.0, num_qubits=num_qubits, encoding=encoding)
        x = x_axis.axis_values
        expected_output_state = np.exp(1j * coef * x**3) * psi.data

        assert np.allclose(output_state.data, expected_output_state)
