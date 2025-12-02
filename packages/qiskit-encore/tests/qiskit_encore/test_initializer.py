"""Unit tests for initializer.py."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_encore.initializer import (
    ideal_state_de_initializer,
    ideal_state_initializer,
    kernel_based_initializer,
    kernel_based_initializer_operator,
)
from qiskit_pytest_helper.constants import FIDELITY_TOLERANCE
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


class TestIdealStateInitializer:
    """Tests for ideal_state_initializer and ideal_state_de_initializer."""

    @given(valid_qiskit_statevector())
    def test_ideal_state_initializer_no_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_initializer function without transpilation of the state initializer."""
        prep_circuit = ideal_state_initializer(statevector)
        generated_state = Statevector(prep_circuit)
        assert state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector())
    def test_ideal_state_initializer_with_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_initializer function with transpilation of the state initializer."""
        prep_circuit = ideal_state_initializer(statevector)
        generated_state = Statevector(prep_circuit)
        assert state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector())
    def test_ideal_state_de_initializer_no_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_de_initializer function without transpilation of the state de-initializer."""
        circuit = QuantumCircuit(statevector.num_qubits)
        circuit.initialize(statevector.data.tolist(), circuit.qubits)
        prep_circuit = ideal_state_de_initializer(statevector)
        circuit.compose(prep_circuit, circuit.qubits, inplace=True)
        de_initialized_state = Statevector(circuit)
        assert abs(de_initialized_state.data[0]) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector())
    def test_ideal_state_de_initializer_with_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_de_initializer function with transpilation of the state de-initializer."""
        circuit = QuantumCircuit(statevector.num_qubits)
        circuit.initialize(statevector.data.tolist(), circuit.qubits)
        prep_circuit = ideal_state_de_initializer(statevector)
        circuit.compose(prep_circuit, circuit.qubits, inplace=True)
        de_initialized_state = Statevector(circuit)
        if abs(de_initialized_state.data[0]) < 1.0 - FIDELITY_TOLERANCE:
            circuit.draw()
        assert abs(de_initialized_state.data[0]) >= 1.0 - FIDELITY_TOLERANCE


class TestKernelBasedInitializer:
    """Tests for kernel_based_initializer."""

    @given(valid_qiskit_statevector())
    def test_kernel_based_initializer(self, statevector: Statevector) -> None:
        """Test the kernel_based_initializer function."""

        prep_circuit = kernel_based_initializer(statevector)
        generated_state = Statevector(prep_circuit)
        assert state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector())
    def test_kernel_based_initializer_operator(self, statevector: Statevector) -> None:
        """Test the kernel_based_initializer_operator function."""

        operator = kernel_based_initializer_operator(statevector)
        assert statevector.num_qubits is not None  # FIXME: stupid qiskit
        zero_state = Statevector.from_label("0" * statevector.num_qubits)
        generated_state = zero_state.evolve(operator)
        assert state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE
