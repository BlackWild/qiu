"""A quantum state together with the way it is prepared in a circuit."""

from dataclasses import dataclass
from functools import cached_property

import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

from qiu_quantum_computing.state_preparation import state_preparation_circuit


@dataclass(frozen=True)
class PreparableState:
    """An immutable quantum state, preparable from the all-zero state by a circuit.

    The preparation circuits are built on first access and cached. Since the state
    and the method cannot change, the cached circuits always prepare the state.
    """

    statevector: Statevector
    """The normalized state. Given amplitudes are converted to a statevector."""
    method: SynthesisMethod = SynthesisMethod.GATE
    """How the preparation is represented in the circuits."""

    def __init__(
        self,
        statevector: Statevector | npt.ArrayLike,
        method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the preparable state.

        Args:
            statevector: The normalized state, as a Qiskit statevector or its
                amplitudes. It is copied.
            method: How the preparation is represented in the circuits, see
                `state_preparation_circuit`.
        """
        object.__setattr__(self, "statevector", validated_statevector(statevector))
        object.__setattr__(self, "method", SynthesisMethod(method))

    @property
    def num_qubits(self) -> int:
        """The number of qubits of the state."""
        num_qubits = self.statevector.num_qubits
        assert num_qubits is not None  # guaranteed by the validation
        return num_qubits

    @cached_property
    def circuit(self) -> QuantumCircuit:
        """The circuit mapping the all-zero state to the state."""
        return state_preparation_circuit(self.statevector, method=self.method)

    @cached_property
    def inverse_circuit(self) -> QuantumCircuit:
        """The circuit mapping the state to the all-zero state."""
        return state_preparation_circuit(
            self.statevector, method=self.method, inverse=True
        )
