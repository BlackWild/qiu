"""Circuits preparing a quantum state from the all-zero state."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector

from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_encore.uniformly_controlled_rotation import uniformly_controlled_rotation


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


def state_preparation_circuit(
    state: Statevector | npt.ArrayLike,
    *,
    method: SynthesisMethod = SynthesisMethod.GATE,
    inverse: bool = False,
) -> QuantumCircuit:
    """Return a circuit preparing the state from the all-zero state.

    The circuit implements a unitary `U` with `U|0...0> = |state>` exactly, including
    the global phase. Its inverse, with `inverse=True`, maps the state back to the
    all-zero state.

    Args:
        state: The normalized state to prepare.
        method: How the preparation is represented in the circuit:
            - `GATE` (default): a single Qiskit `StatePreparation` gate, synthesized
              by Qiskit when transpiling. Note that Qiskit's synthesis prepares
              wrong states for some inputs (qiskit 2.2 to 2.5), see
              `decomposed_state_preparation`.
            - `DECOMPOSED`: elementary `ry`, `rz` and `cx` gates, synthesized by
              `decomposed_state_preparation`.
            - `DENSE`: a single unitary gate of the matrix of
              `state_preparation_unitary`.
        inverse: If True, return the circuit mapping the state to the all-zero state.

    Returns:
        The state preparation circuit.
    """
    statevector = validated_statevector(state)
    method = SynthesisMethod(method)

    if method == SynthesisMethod.DECOMPOSED:
        circuit = decomposed_state_preparation(statevector)
        return circuit.inverse() if inverse else circuit

    circuit = QuantumCircuit(statevector.num_qubits, name="state_preparation")
    if method == SynthesisMethod.GATE:
        circuit.append(StatePreparation(statevector, inverse=inverse), circuit.qubits)
    elif method == SynthesisMethod.DENSE:
        unitary = state_preparation_unitary(statevector)
        circuit.unitary(
            unitary.adjoint() if inverse else unitary,
            circuit.qubits,
            label="state_preparation_dg" if inverse else "state_preparation",
        )
    else:
        raise NotImplementedError(f"Unsupported synthesis method: {method.value!r}")
    return circuit


def decomposed_state_preparation(state: Statevector | npt.ArrayLike) -> QuantumCircuit:
    """Return a circuit of elementary gates preparing the state from the all-zero state.

    The synthesis of Möttönen et al., "Transformation of quantum states using
    uniformly controlled rotations" (2005), first sets the magnitudes of the
    amplitudes with uniformly controlled `ry` rotations, from the most significant
    qubit down, and then their phases with uniformly controlled `rz` rotations and
    the global phase of the circuit. It uses at most `2**(n+1) - 4` CNOT gates for
    `n` qubits, and skips the phase stage for real non-negative states.

    All angles are computed with `arctan2` and sums, without eigendecompositions,
    unlike Qiskit's `StatePreparation`, whose isometry synthesis prepares wrong
    states when two of its intermediate single-qubit gates are close but not equal.

    Args:
        state: The normalized state to prepare.

    Returns:
        The circuit of `ry`, `rz` and `cx` gates.
    """
    statevector = validated_statevector(state)
    num_qubits = statevector.num_qubits
    assert num_qubits is not None  # guaranteed by the validation
    amplitudes = statevector.data

    circuit = QuantumCircuit(num_qubits, name="state_preparation")

    # Magnitudes: rotate each target qubit t, conditioned on the more significant
    # qubits t+1, ..., n-1, into the conditional distribution of its bit.
    probabilities = np.abs(amplitudes) ** 2
    for target in reversed(range(num_qubits)):
        num_controls = num_qubits - 1 - target
        marginals = probabilities.reshape(2**num_controls, 2, 2**target).sum(axis=2)
        angles = 2 * np.arctan2(np.sqrt(marginals[:, 1]), np.sqrt(marginals[:, 0]))
        circuit.compose(
            uniformly_controlled_rotation("y", angles),
            qubits=range(target, num_qubits),
            inplace=True,
        )

    # Phases: split the phases of each pair of amplitudes differing in the bit of
    # the target t into their difference, applied by an rz rotation, and their mean,
    # left to the more significant qubits and finally to the global phase.
    phases = np.where(probabilities > 0, np.angle(amplitudes), 0.0)
    for target in range(num_qubits):
        pairs = phases.reshape(-1, 2)
        circuit.compose(
            uniformly_controlled_rotation("z", pairs[:, 1] - pairs[:, 0]),
            qubits=range(target, num_qubits),
            inplace=True,
        )
        phases = pairs.mean(axis=1)
    circuit.global_phase = phases[0]

    return circuit


def state_preparation_unitary(state: Statevector | npt.ArrayLike) -> Operator:
    """Return a dense unitary whose first column is the state.

    The unitary is `-e^(i theta) H`, where `H` is the Householder reflection swapping
    `|0...0>` and `-e^(-i theta)|state>`, with `theta` the phase of the first
    amplitude. It is exact and numerically stable, and takes `O(4**n)` memory.

    Args:
        state: The normalized state to prepare.

    Returns:
        The unitary operator `U` with `U|0...0> = |state>`.
    """
    amplitudes = validated_statevector(state).data
    theta = np.angle(amplitudes[0]) if amplitudes[0] != 0 else 0.0

    # rotated such that the first amplitude is real and non-negative
    rotated = np.exp(-1j * theta) * amplitudes
    rotated[0] = abs(amplitudes[0])

    # reflecting along u = rotated + |0>, whose norm is at least 1 for stability
    u = rotated.copy()
    u[0] += 1
    reflection = np.eye(u.size) - 2 * np.outer(u, u.conj()) / np.vdot(u, u).real

    return Operator(-np.exp(1j * theta) * reflection)
