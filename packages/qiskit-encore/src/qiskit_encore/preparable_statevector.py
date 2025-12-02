"""A Statevector that can be prepared by a quantum circuit."""

from collections.abc import Callable
from functools import cached_property

import numpy as np
import numpy.typing as npt
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_encore.helper_types import InitializerType
from qiskit_encore.initializer import (
    big_unitary_matrix_state_de_initializer,
    big_unitary_matrix_state_initializer,
    generic_qiskit_state_initializer,
    ideal_state_initializer,
    kernel_based_initializer,
)


class PreparableStatevector(Statevector):
    """A Statevector that can be prepared by a quantum circuit."""

    initializer_generator: InitializerType
    """The function that generates the initializer circuit."""

    de_initializer_generator: InitializerType | None

    @property
    def num_qubits(self) -> int:
        """The number of qubits in the quantum state."""
        return int(np.log2(len(self.data)))

    def __init__(
        self,
        data: npt.ArrayLike,  # TODO: data: QiskitStatevectorDataType ?????,
        initializer_generator: InitializerType,
        de_initializer_generator: InitializerType | None = None,
        normalize: bool = False,
    ) -> None:
        """Initializes the PreparableStatevector with the given wavefunction."""
        data = np.asarray(data, dtype=np.complex128)
        if normalize:
            data = data / np.linalg.norm(data)
        super().__init__(data)
        self.initializer_generator = initializer_generator
        self.de_initializer_generator = de_initializer_generator

    @cached_property
    def initializer_circuit(self) -> QuantumCircuit:
        """The initializer circuit for the quantum state."""
        return self.initializer_generator(self)

    @cached_property
    def de_initializer_circuit(self) -> QuantumCircuit:
        """The de-initializer circuit for the quantum state."""
        if self.de_initializer_generator is not None:
            return self.de_initializer_generator(self)
        else:
            return self.initializer_circuit.inverse()


class IdeallyPreparableStatevector(PreparableStatevector):
    """A PreparableStatevector that uses the ideal state initializer."""

    def __init__(
        self,
        data: npt.ArrayLike,
        normalize: bool = False,
    ) -> None:
        """Initializes the IdealPreparableStatevector with the given wavefunction."""
        super().__init__(
            data,
            initializer_generator=ideal_state_initializer,
            normalize=normalize,
        )

    @classmethod
    def from_statevector(
        cls, statevector: Statevector
    ) -> "IdeallyPreparableStatevector":
        """Creates an IdealPreparableStatevector from a Qiskit Statevector."""
        return cls(statevector.data, normalize=False)


class GenericQiskitPreparableStatevector(PreparableStatevector):
    """A PreparableStatevector that uses arbitrary Qiskit circuits for initialization."""

    def __init__(
        self,
        data: npt.ArrayLike,
        normalize: bool = False,
    ) -> None:
        """Initializes the GenericQiskitPreparableStatevector with the given wavefunction and circuits."""
        super().__init__(
            data,
            initializer_generator=generic_qiskit_state_initializer,
            normalize=normalize,
        )

    @classmethod
    def from_statevector(
        cls,
        statevector: Statevector,
    ) -> "GenericQiskitPreparableStatevector":
        """Creates a GenericQiskitPreparableStatevector from a Qiskit Statevector and circuits."""
        return cls(
            statevector.data,
            normalize=False,
        )


class BigUnitaryPreparableStatevector(PreparableStatevector):
    def __init__(
        self,
        data: npt.ArrayLike,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            data,
            initializer_generator=big_unitary_matrix_state_initializer,
            normalize=normalize,
        )

    @classmethod
    def from_statevector(
        cls, statevector: Statevector
    ) -> "BigUnitaryPreparableStatevector":
        """Creates an IdealPreparableStatevector from a Qiskit Statevector."""
        return cls(statevector.data, normalize=False)


class KernelBasedPreparableStatevector(PreparableStatevector):
    """A PreparableStatevector that uses kernel-based method for initialization."""

    def __init__(
        self,
        data: npt.ArrayLike,
        normalize: bool = False,
    ) -> None:
        """Initializes the KernelBasedPreparableStatevector with the given wavefunction."""
        super().__init__(
            data,
            initializer_generator=kernel_based_initializer,
            normalize=normalize,
        )

    @classmethod
    def from_statevector(
        cls, statevector: Statevector
    ) -> "KernelBasedPreparableStatevector":
        """Creates a KernelBasedPreparableStatevector from a Qiskit Statevector."""
        return cls(statevector.data, normalize=False)
