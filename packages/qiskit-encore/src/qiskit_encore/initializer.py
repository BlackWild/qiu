"""Initializers."""

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation
from qiskit.quantum_info import Operator, Statevector

from qiskit_encore.state_preparation import StatePreparationCircuit


def ideal_state_initializer(state: Statevector) -> QuantumCircuit:
    """Generate a quantum circuit that initializes a quantum state to the given state.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        QuantumCircuit: A quantum circuit that initializes a quantum state to the given state.
    """
    print("Using ideal state initializer.")  # TODO: remove this
    circ = StatePreparationCircuit(state)
    return circ


def ideal_state_de_initializer(state: Statevector) -> QuantumCircuit:
    """Generate a quantum circuit that de-initializes a quantum state from the given state.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        QuantumCircuit: A quantum circuit that de-initializes a quantum state from the given state.
    """
    circ = StatePreparationCircuit(state, inverse=True)

    return circ


def generic_qiskit_state_initializer(state: Statevector) -> QuantumCircuit:
    print("Using generic qiskit state initializer.")  # TODO: remove this
    qiskit_prep = QiskitStatePreparation(state)
    init_circ = QuantumCircuit(qiskit_prep.num_qubits)
    init_circ.append(qiskit_prep, init_circ.qubits)
    return init_circ


def big_unitary_matrix_state_initializer(state: Statevector) -> QuantumCircuit:
    print("Using big matrix state initializer.")  # TODO: remove this
    qiskit_prep = QiskitStatePreparation(state)
    init_op = Operator(qiskit_prep)
    init_gate = init_op.to_instruction()
    init_circ = QuantumCircuit(init_gate.num_qubits)
    init_circ.append(init_gate, init_circ.qubits)
    return init_circ


def big_unitary_matrix_state_de_initializer(state: Statevector) -> QuantumCircuit:
    print("Using big matrix state de-initializer.")  # TODO: remove this
    qiskit_prep = QiskitStatePreparation(state)
    inv = qiskit_prep.inverse()
    de_init_op = Operator(inv)
    de_init_gate = de_init_op.to_instruction()
    de_init_circ = QuantumCircuit(de_init_gate.num_qubits)
    de_init_circ.append(de_init_gate, de_init_circ.qubits)
    return de_init_circ


# # def mps_based_initializer(state: Statevector) -> Gate:
