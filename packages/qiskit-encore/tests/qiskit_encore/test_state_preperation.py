"""Unit tests for state_preparation.py."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_encore.state_preparation import StatePreparationCircuit
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


@given(
    state=valid_qiskit_statevector(),
    transpilation=st.booleans(),
)
def test_state_preparation(state: Statevector, transpilation: bool) -> None:
    """Test the StatePreparationCircuit."""

    assert isinstance(state.num_qubits, int)
    prep_circuit = StatePreparationCircuit(state, normalize=False)

    assert isinstance(prep_circuit, QuantumCircuit)

    operator = Operator(prep_circuit)
    assert operator.is_unitary()
    assert operator.dim == (2**state.num_qubits, 2**state.num_qubits)
    assert np.allclose(operator.data[:, 0], state.data)  # type: ignore

    backend = AerSimulator()
    circuit_to_simulate = (
        transpile(prep_circuit, backend) if transpilation else prep_circuit
    )
    job = backend.run(circuit_to_simulate, shots=1)
    result = job.result()

    assert result.success


@given(
    state=valid_qiskit_statevector(),
    transpilation=st.booleans(),
)
def test_state_de_preparation(state: Statevector, transpilation: bool) -> None:
    """Test the StatePreparationCircuit in inverse mode."""

    assert isinstance(state.num_qubits, int)
    prep_circuit = StatePreparationCircuit(state, inverse=True, normalize=False)

    assert isinstance(prep_circuit, QuantumCircuit)

    operator = Operator(prep_circuit)
    assert operator.is_unitary()
    assert operator.dim == (2**state.num_qubits, 2**state.num_qubits)
    assert np.allclose(operator.data[0, :], state.data.conj())  # type: ignore

    backend = AerSimulator()
    circuit_to_simulate = (
        transpile(prep_circuit, backend) if transpilation else prep_circuit
    )
    job = backend.run(circuit_to_simulate, shots=1)
    result = job.result()

    assert result.success
