"""Initializers."""

from typing import Callable

from qiskit.circuit import Gate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

InitializerType = Callable[[Statevector], Gate]
"""Type alias for a quantum state initializer function."""


def ideal_state_initializer(state: Statevector) -> Gate:
    """Generate a quantum circuit that initializes a quantum state to the given state.

    Args:
        state (Statevector): The target quantum state as a state vector.

    Returns:
        Gate: A quantum gate that initializes a quantum state to the given state.
    """
    state_preparation = StatePreparation(state)
    return state_preparation


# def mps_based_initializer(state: Statevector) -> Gate:
