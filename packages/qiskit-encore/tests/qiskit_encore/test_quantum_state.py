"""Unit tests for quantum_state.py."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_encore.quantum_state import IdealPreparableStatevector

ERROR_TOLERANCE = 1e-10

# TODO: create a custom strategy for generating valid quantum states
# TODO: decouple the tests to three -> basics, initialization, and de-initialization


class TestIdealPreparableStatevector:
    """Tests for IdealPreparableStatevector."""

    @given(
        arrays(
            dtype=np.complex128,
            shape=st.sampled_from([2, 4, 8, 16, 32]),
            elements=st.complex_numbers(
                min_magnitude=0.01,
                max_magnitude=1000,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    )
    def test_everything(self, data: npt.NDArray[np.complex128]):
        """A test to make sure the class can be instantiated, normalized, converted to a gate, and de-initialized."""

        state = IdealPreparableStatevector(data, normalize=True)

        assert isinstance(state, Statevector)
        assert state.data.shape == data.shape
        assert np.allclose(state.data, data / np.linalg.norm(data))
        assert np.isclose(np.linalg.norm(state.data), 1.0)

        output_state = Statevector(state.initializer_gate)
        assert np.allclose(output_state.data, state.data)

        circuit = QuantumCircuit(state.num_qubits)
        circuit.initialize(state.data.tolist(), circuit.qubits)
        circuit.append(state.de_initializer_gate, circuit.qubits)

        de_initialized_state = Statevector(circuit)
        expected_state = np.zeros_like(data)
        expected_state[0] = 1.0
        assert np.allclose(
            de_initialized_state.data, expected_state, atol=ERROR_TOLERANCE
        )
