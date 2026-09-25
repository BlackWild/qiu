"""Qiskit statevectors, validated as the states of qubits."""

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import Statevector


def validated_statevector(state: Statevector | npt.ArrayLike) -> Statevector:
    """Return a copy of the state as a normalized statevector of at least one qubit.

    The normalization is checked with `Statevector.is_valid`, i.e. up to Qiskit's
    tolerances `Statevector.atol` and `Statevector.rtol`.

    Args:
        state: The state, as a Qiskit statevector or its amplitudes.

    Returns:
        The state as a Qiskit statevector.

    Raises:
        ValueError: If the state is not a state of at least one qubit, i.e. its
            dimension is not a power of 2 larger than 1, or if it is not normalized.
    """
    data = state.data if isinstance(state, Statevector) else state
    statevector = Statevector(np.array(data, dtype=np.complex128))

    if statevector.num_qubits is None or statevector.num_qubits < 1:
        raise ValueError(
            f"The state must be a state of at least one qubit, got dimension "
            f"{statevector.dim}."
        )
    if not statevector.is_valid():
        norm = np.linalg.norm(statevector.data)
        raise ValueError(f"The state must be normalized, got norm {norm}.")

    return statevector
