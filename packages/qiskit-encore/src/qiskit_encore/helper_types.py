"""Helper types for Qiskit Encore."""

import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

QiskitStatevectorDataType = (
    np.ndarray | list | Statevector | Operator | QuantumCircuit | Instruction
)
