"""Typed helpers to inspect circuits in unit tests.

Qiskit's annotations of `Operator.data` and `QuantumCircuit.count_ops` are too loose
or wrong for type checkers, so tests use these helpers instead.
"""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


def unitary_matrix(circuit: QuantumCircuit | Operator) -> npt.NDArray[np.complex128]:
    """Return the dense unitary matrix of a circuit or operator."""
    operator = circuit if isinstance(circuit, Operator) else Operator(circuit)
    return np.asarray(operator.data, dtype=np.complex128)


def gate_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Return the number of gates of each name in the circuit."""
    return {str(name): count for name, count in circuit.count_ops().items()}
