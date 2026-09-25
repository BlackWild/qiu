"""Approximate state preparation with layers of MPS gates.

Each layer prepares the MPS of bond dimension 2 approximating the part of the state that
the layers before it leave: with the residual `|r> = U_k^dagger ... U_1^dagger |state>`
of `k` layers, the next layer `U_(k+1)` prepares the bond-2 approximation of `|r>`. The
circuit `U_1 ... U_k` then prepares `|state>` up to the error `|| |r> - |0...0> ||`,
global phase included.

States of up to 3 qubits have bond dimension at most 2, so one layer prepares them
exactly. For more qubits, the error decreases with the number of layers, though not
always monotonically.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiu_qiskit_encore.statevector import validated_statevector
from qiu_quantum_computing.state_preparation import state_preparation_unitary

from qiu_mps_initializer.mps import bond2_mps_approximation, mps_layer


@dataclass(frozen=True)
class MPSStatePreparation:
    """A circuit preparing a state approximately from `|0...0>`, with its error."""

    circuit: QuantumCircuit
    """The circuit, its layers applied in order."""
    layers: tuple[QuantumCircuit, ...]
    """The layers of the circuit, in the order they are applied."""
    error: float
    """The distance of the prepared state to the target, global phase included."""
    converged: bool
    """Whether the prepared state reached the target within the tolerance."""

    @property
    def num_layers(self) -> int:
        """The number of layers."""
        return len(self.layers)


def mps_state_preparation(
    state: Statevector | npt.ArrayLike,
    max_layers: int,
    tolerance: float | None = None,
) -> MPSStatePreparation:
    """Return the circuit of at most `max_layers` MPS layers preparing a state.

    Layers are added until the prepared state reaches the target, or `max_layers`.

    Args:
        state: The normalized state, of at least one qubit.
        max_layers: The maximum number of layers, at least 1.
        tolerance: The distance to the target, global phase included, below which
            the prepared state reaches it. By default, it reaches it when it equals
            the target as a Qiskit `Statevector`, i.e. up to Qiskit's tolerances.

    Returns:
        The circuit, its layers and the error of the prepared state.

    Raises:
        ValueError: If `max_layers` is smaller than 1, or if the state is not a
            normalized state of at least one qubit.
    """
    if max_layers < 1:
        raise ValueError(f"The max_layers must be at least 1, got {max_layers}.")
    target = validated_statevector(state)
    num_qubits = target.num_qubits
    assert num_qubits is not None

    zero_state = Statevector.from_int(0, 2**num_qubits)

    def reached(residual: Statevector) -> bool:
        if tolerance is None:
            return residual == zero_state
        return float(np.linalg.norm(residual.data - zero_state.data)) < tolerance

    layers: list[QuantumCircuit] = []
    residual = target
    while not reached(residual) and len(layers) < max_layers:
        layer = (
            _single_qubit_layer(residual)
            if num_qubits == 1
            else mps_layer(bond2_mps_approximation(residual))
        )
        residual = residual.evolve(Operator(layer).adjoint())
        layers.append(layer)

    circuit = QuantumCircuit(num_qubits, name="mps_state_preparation")
    # the residual is disentangled by the last layer first, so it is applied first
    for layer in reversed(layers):
        circuit.compose(layer, inplace=True)

    return MPSStatePreparation(
        circuit=circuit,
        layers=tuple(reversed(layers)),
        error=float(np.linalg.norm(residual.data - zero_state.data)),
        converged=reached(residual),
    )


def _single_qubit_layer(state: Statevector) -> QuantumCircuit:
    """Return the layer preparing a single-qubit state exactly."""
    circuit = QuantumCircuit(1, name="mps_layer")
    circuit.unitary(state_preparation_unitary(state), [0], label="G0")
    return circuit
