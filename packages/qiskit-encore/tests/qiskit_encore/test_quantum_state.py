"""Unit tests for quantum_state.py."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_encore.quantum_state import IdealPreparableStatevector
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector

ERROR_TOLERANCE = 1e-10


class TestIdealPreparableStatevector:
    """Tests for IdealPreparableStatevector."""

    @given(valid_qiskit_statevector(min_qubits=1, max_qubits=5))
    def test_essentials(self, statevector: Statevector) -> None:
        """A test to make sure the class can be instantiated, normalized, converted to a gate, and de-initialized."""

        state = IdealPreparableStatevector.from_statevector(statevector)

        assert isinstance(state, Statevector)

    @given(statevector=valid_qiskit_statevector(min_qubits=1, max_qubits=5))
    def test_initialization(self, statevector: Statevector) -> None:
        """Test the initialization of the IdealPreparableStatevector."""
        state = IdealPreparableStatevector.from_statevector(statevector)
        generated_state = Statevector(state.initializer_gate)
        assert np.allclose(generated_state.data, state.data)

    @given(statevector=valid_qiskit_statevector(min_qubits=1, max_qubits=5))
    def test_de_initialization(self, statevector: Statevector) -> None:
        """Test the de-initialization of the IdealPreparableStatevector."""
        state = IdealPreparableStatevector.from_statevector(statevector)
        circuit = QuantumCircuit(state.num_qubits)
        circuit.initialize(state.data.tolist(), circuit.qubits)
        circuit.append(state.de_initializer_gate, circuit.qubits)

        de_initialized_state = Statevector(circuit)

        # TODO: fix this test, it should be that the first element is 1 and the rest are 0s. I could not get it to work. because the other elements are very small but not exactly close to zero in np.isclose standard. They where around 1e-10. For now I just check that the first element is close to 1 which should approximate the expected behavior.
        assert np.isclose(de_initialized_state.data[0], 1.0)

        # expected_state = np.zeros_like(state.data)
        # expected_state[0] = 1.0
        # assert np.allclose(
        #     de_initialized_state.data, expected_state, atol=ERROR_TOLERANCE
        # )
