"""Module for quantum states that can be prepared by quantum circuits."""

from functools import cached_property
import numpy as np
import numpy.typing as npt

from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector

from qiskit_encore.helper_types import QiskitStatevectorDataType
from qiskit_encore.initializer import InitializerType, ideal_state_initializer


class PreparableStatevector(Statevector):
    """A Statevector that can be prepared by a quantum circuit."""

    initializer_generator: InitializerType
    """The function that generates the initializer circuit."""

    de_initializer_generator: InitializerType | None

    def __init__(
        self,
        data: npt.ArrayLike, # TODO: data: QiskitStatevectorDataType ?????,
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
    def initializer_gate(self) -> Gate:
        """The initializer gate for the quantum state."""
        return self.initializer_generator(self)

    @cached_property
    def de_initializer_gate(self) -> Gate:
        """The de-initializer gate for the quantum state."""
        if self.de_initializer_generator is not None:
            return self.de_initializer_generator(self)
        else:
            return self.initializer_gate.inverse()  # type: ignore


class IdealPreparableStatevector(PreparableStatevector):
    """A PreparableStatevector that uses the ideal state initializer."""

    def __init__(
        self,
        data: npt.ArrayLike,
        normalize: bool = False,
    ) -> None:
        """Initializes the IdealPreparableStatevector with the given wavefunction."""
        super().__init__(data, initializer_generator=ideal_state_initializer, normalize=normalize)
