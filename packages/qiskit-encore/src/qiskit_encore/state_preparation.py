"""State preparation circuit with decomposition to elementary gates."""

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation
from qiskit.quantum_info import Statevector


class StatePreparationCircuit(QuantumCircuit):
    """State preparation circuit which is decomposed into elementary gates by default."""

    def __init__(
        self,
        params: str | list | int | Statevector,
        num_qubits: int | None = None,
        inverse: bool = False,
        label: str | None = None,
        normalize: bool = False,
    ):
        """Create a state preparation circuit."""
        qiskit_prep = QiskitStatePreparation(
            params,
            num_qubits=num_qubits,
            inverse=inverse,
            label=label,
            normalize=normalize,
        )

        # Decompose twice to ensure all composite gates are broken down to elementary gates. This has been confirmed to work by manual inspection.
        definition: QuantumCircuit = qiskit_prep.definition  # type: ignore[assignment]
        decomposed_circuit = definition.decompose(reps=2)

        super().__init__(decomposed_circuit.num_qubits)
        self.compose(decomposed_circuit, inplace=True)
