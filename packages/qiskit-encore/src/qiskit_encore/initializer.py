"""Initializers."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

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


def ideal_state_init_de_init_pair(
    state: Statevector,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Generate a pair of quantum circuits that initialize and de-initialize a quantum state.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        tuple[QuantumCircuit, QuantumCircuit]: A tuple containing two quantum circuits: the first initializes a quantum state to the given state, and the second de-initializes a quantum state from the given state.
    """
    initializer = ideal_state_initializer(state)
    de_initializer = initializer.inverse()
    return (
        initializer,
        de_initializer,
    )


# # def mps_based_initializer(state: Statevector) -> QuantumCircuit:
