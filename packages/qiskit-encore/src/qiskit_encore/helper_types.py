"""Helper types for Qiskit Encore."""

from collections.abc import Callable

import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

QiskitStatevectorDataType = (
    np.ndarray | list | Statevector | Operator | QuantumCircuit | Instruction
)

InitializerType = Callable[[Statevector], QuantumCircuit]
"""Type alias for a quantum state initializer function."""
