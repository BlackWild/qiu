"""Collection of tools to represent quantum signals."""

from functools import cached_property

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import Statevector

from qiskit_encore.helper_types import QiskitStatevectorDataType
from qiskit_encore.quantum_state import (
    IdealPreparableStatevector,
    PreparableStatevector,
)


class GenericQuantumSignal:
    """A generic quantum signal having no specific structure imposed on it. It is used to be exported to other signal types such as a LinearSignal, QuadraticSignal, etc."""

    _data: npt.NDArray[np.float64]
    """The data of the quantum signal."""

    def __init__(self, data: npt.ArrayLike) -> None:
        """Initializes the GenericQuantumSignal with the given data."""
        self._data = np.asarray(data, dtype=np.float64)

    def to_quadratic(self) -> "QuadraticQuantumSignal":
        """Converts the generic quantum signal to a quadratic quantum signal."""
        return QuadraticQuantumSignal.from_data(self._data)
    
    def __mul__(self, other: int | float) -> "GenericQuantumSignal":
        """Defines the multiplication of the GenericQuantumSignal with a scalar."""
        if isinstance(other, int | float):  # type: ignore
            new_data = self._data * other
            return GenericQuantumSignal(new_data)
        else:
            raise ValueError("Multiplication is only defined for scalars.")
        
    def __add__(self, other: "GenericQuantumSignal") -> "GenericQuantumSignal":
        """Defines the addition of two GenericQuantumSignal objects."""
        if isinstance(other, GenericQuantumSignal):
            if self._data.shape != other._data.shape:
                raise ValueError("Signals must have the same shape to be added.")
            new_data = self._data + other._data
            return GenericQuantumSignal(new_data)
        else:
            raise ValueError("Addition is only defined between GenericQuantumSignal objects.")


class QuadraticQuantumSignal():
    """A quadratic quantum signal. Also called an intensity signal."""

    alpha: float
    """The alpha parameter of the quadratic quantum signal."""
    statevector: Statevector
    """Returns the statevector representation of the quantum signal."""

    def __init__(self, alpha: float, statevector: Statevector) -> None:
        """Initializes the QuadraticQuantumSignal."""
        self.alpha = alpha
        self.statevector = statevector

    @classmethod
    def from_data(cls, data: npt.ArrayLike) -> "QuadraticQuantumSignal":
        """Creates a QuadraticQuantumSignal from raw data."""
        alpha, state = extract_alpha_and_state_from_generic_signal(np.asarray(data), power=2)
        return cls(alpha, state)
    

def extract_alpha_and_state_from_generic_signal(
    signal: npt.NDArray[np.float64], power: int
) -> tuple[float, Statevector]:
    """Extracts the alpha and the state from the generic signal."""

    # check if all elements of signal have the same sign
    if not all(
        [np.sign(signal[0]) == np.sign(signal[i]) or np.isclose(signal[i], 0) for i in range(len(signal))]
    ):
        raise ValueError("All elements of the signal vector must have the same sign.")

    # normalization factor
    alpha = np.sum(signal)

    # normalized signal
    normalized_signal = signal / alpha

    # corresponding wavefunction
    state_data = normalized_signal ** (1 / power)

    state = Statevector(state_data)

    return alpha, state