"""Typed helpers to inspect and transpile circuits in unit tests.

Qiskit's annotations of `Operator.data` and `QuantumCircuit.count_ops` are too loose
or wrong for type checkers, so tests use these helpers instead.
"""

import numpy as np
import numpy.typing as npt
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator


def unitary_matrix(circuit: QuantumCircuit | Operator) -> npt.NDArray[np.complex128]:
    """Return the dense unitary matrix of a circuit or operator."""
    operator = circuit if isinstance(circuit, Operator) else Operator(circuit)
    return np.asarray(operator.data, dtype=np.complex128)


def gate_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Return the number of gates of each name in the circuit."""
    return {str(name): count for name, count in circuit.count_ops().items()}


EXACT_OPTIMIZATION_LEVEL = 1
"""The highest transpiler optimization level that keeps circuits exact.

From level 2, the transpiler removes gates it deems equivalent to the identity, e.g.
rotations by angles of `1e-7`, which changes the amplitudes by as much.
"""


def transpile_exactly(circuit: QuantumCircuit, backend: BackendV2) -> QuantumCircuit:
    """Transpile the circuit for the backend without approximating it."""
    return transpile(circuit, backend, optimization_level=EXACT_OPTIMIZATION_LEVEL)
