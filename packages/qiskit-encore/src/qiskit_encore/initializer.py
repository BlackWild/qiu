"""Initializers."""

from collections.abc import Callable

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

InitializerType = Callable[[Statevector], Gate]
"""Type alias for a quantum state initializer function."""


def ideal_state_initializer(state: Statevector, force_transpile: bool = False) -> Gate:
    """Generate a quantum circuit that initializes a quantum state to the given state.

    Args:
        state (Statevector): The target quantum state as a state vector.
        force_transpile (bool): If True, transpile the circuit to use only 'cx' and 'u' gates. Default is False.

    Returns:
        Gate: A quantum gate that initializes a quantum state to the given state.
    """
    state_preparation = StatePreparation(state)
    if force_transpile:
        qc = QuantumCircuit(state.num_qubits)
        qc.append(state_preparation, qc.qubits)
        return transpile(qc, basis_gates=["cx", "u"]).to_gate()

    return state_preparation


def ideal_state_de_initializer(
    state: Statevector, force_transpile: bool = False
) -> Gate:
    """Generate a quantum circuit that de-initializes a quantum state from the given state.

    Args:
        state (Statevector): The target quantum state as a state vector.
        force_transpile (bool): If True, transpile the circuit to use only 'cx' and 'u' gates. Default is False.

    Returns:
        Gate: A quantum gate that de-initializes a quantum state from the given state.
    """
    state_preparation = StatePreparation(state, inverse=True)
    if force_transpile:
        qc = QuantumCircuit(state.num_qubits)
        qc.append(state_preparation, qc.qubits)
        return transpile(qc, basis_gates=["cx", "u"]).to_gate()
    return state_preparation


# def mps_based_initializer(state: Statevector) -> Gate:
