"""QuantumState."""

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit


class QuantumState:
    """Represents a quantum state."""

    wavefunction: npt.NDArray[np.complex128]
    """The wavefunction of the quantum state."""

    initializer_circuit: QuantumCircuit | None
    """The initializer circuit for the quantum state."""
    de_initializer_circuit: QuantumCircuit | None
    """The de-initializer circuit for the quantum state."""

    def __init__(self, wavefunction: npt.ArrayLike, normalize: bool = False) -> None:
        """Initializes the QuantumState with the given wavefunction.

        Args:
            wavefunction (npt.ArrayLike): The wavefunction of the quantum state.
            normalize (bool, optional): Whether to normalize the wavefunction if it is not normalized. Defaults to False.
        """
        normalization_factor = np.linalg.norm(self.wavefunction)

        if normalize and not np.isclose(normalization_factor, 1.0):
            self.wavefunction = (
                np.array(wavefunction, dtype=np.complex128) / normalization_factor
            )
        else:
            raise ValueError(
                "The provided wavefunction is not normalized. Set `normalize=True` to force the normalization of the wavefunction."
            )

    @property
    def size(self) -> int:
        """The dimension of the quantum state."""
        return self.wavefunction.size

    @property
    def qubits_size(self) -> int:
        """The number of qubits required to represent the quantum state."""
        return np.log2(self.size).astype(int)

    # def generate_mps_initializer_circuit(
    #     self, number_of_layers: int
    # ) -> qiskit.circuit.QuantumCircuit:
    #     """Generates the MPS initializer circuit for the quantum state.

    #     Returns:
    #         QuantumCircuit: The MPS initializer circuit for the quantum state as a qiskit circuit.
    #     """
    #     circuit, _ = multi_layered_circuit_for_non_approximated(
    #         self.wavefunction, max_number_of_layers=number_of_layers
    #     )
    #     return circuit
