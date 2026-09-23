"""Initializers."""

import numpy as np
import scipy
from qiskit.circuit import QuantumCircuit
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
    qiskit_prep = QiskitStatePreparation(state)
    init_circ = QuantumCircuit(qiskit_prep.num_qubits)
    init_circ.append(qiskit_prep, init_circ.qubits)
    return init_circ


def big_unitary_matrix_state_initializer(state: Statevector) -> QuantumCircuit:
    qiskit_prep = QiskitStatePreparation(state)
    init_op = Operator(qiskit_prep)
    init_gate = init_op.to_instruction()
    init_circ = QuantumCircuit(init_gate.num_qubits)
    init_circ.append(init_gate, init_circ.qubits)
    return init_circ


def big_unitary_matrix_state_de_initializer(state: Statevector) -> QuantumCircuit:
    qiskit_prep = QiskitStatePreparation(state)
    inv = qiskit_prep.inverse()
    de_init_op = Operator(inv)
    de_init_gate = de_init_op.to_instruction()
    de_init_circ = QuantumCircuit(de_init_gate.num_qubits)
    de_init_circ.append(de_init_gate, de_init_circ.qubits)
    return de_init_circ


def kernel_based_initializer(state: Statevector) -> QuantumCircuit:
    """Generate a quantum circuit that initializes a quantum state using kernel-based method.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        Gate: A gate that initializes a quantum state to the given state.
    """

    # TODO: clean this up

    print("1")
    data = state.data.reshape((state.dim, 1))
    print("2")

    null_space = scipy.linalg.null_space(data.T)
    print("3")
    matrix = np.column_stack((data, null_space.conjugate()))
    print("4")

    circuit = QuantumCircuit(state.num_qubits)
    print("5")
    gate = Operator(matrix)
    print("6")
    inst = gate.to_instruction()
    print("7")
    circuit.compose(inst, circuit.qubits, inplace=True)
    print("8")
    return circuit


def kernel_based_initializer_operator(state: Statevector) -> Operator:
    """Generate an operator that initializes a quantum state using kernel-based method.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        Operator: An operator that initializes a quantum state to the given state.
    """

    data = state.data.reshape((state.dim, 1))
    null_space = scipy.linalg.null_space(data.T)
    matrix = np.column_stack((data, null_space.conjugate()))
    gate = Operator(matrix)
    return gate


# # def mps_based_initializer(state: Statevector) -> Gate:
