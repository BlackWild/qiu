"""Collection of tools to represent quantum signals."""

from functools import cached_property

import numpy as np
import numpy.typing as npt

from qiskit_signals.helper_types import EncodingType, SignalFunctionType
from qiskit_signals.quantum_axis import GenericAxis

# - MARK: Generic


class GenericQuantumSignal:
    """A generic quantum signal having no specific structure imposed on it."""

    axis: GenericAxis
    signal_function: SignalFunctionType

    def __init__(self, axis: GenericAxis, signal_function: SignalFunctionType) -> None:
        """Initializes the GenericQuantumSignal with the given axis and signal function."""
        self.axis = axis
        self.signal_function = signal_function

    @cached_property
    def data(self) -> npt.NDArray[np.float64]:
        """Returns the data of the quantum signal."""
        return self.signal_function(self.axis.axis_values)

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the signal."""
        return self.axis.num_qubits

    @property
    def dimension(self) -> int:
        """Returns the dimension of the signal."""
        return self.axis.dimension

    @property
    def normalized_data(self) -> npt.NDArray[np.float64]:
        """Returns the normalized data of the quantum signal."""
        data = self.data
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        return data / norm


class PolynomialQuantumSignal:
    """A polynomial quantum signal."""

    axis: GenericAxis
    """The axis of the quantum signal."""
    alpha: float
    """The coefficient parameter of the polynomial quantum signal."""
    power: int
    """The power of the polynomial quantum signal."""

    def __init__(self, axis: GenericAxis, alpha: float, power: int) -> None:
        """Initializes the PolynomialQuantumSignal."""
        self.axis = axis
        self.alpha = alpha
        self.power = power

    @cached_property
    def data(self) -> npt.NDArray[np.float64]:
        """Returns the data of the quantum signal."""
        return self.alpha * self.axis.axis_values**self.power

    def to_generic(self) -> GenericQuantumSignal:
        """Converts the polynomial quantum signal to a generic quantum signal."""
        return GenericQuantumSignal(
            axis=self.axis, signal_function=lambda x: self.alpha * x**self.power
        )

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the signal."""
        return self.axis.num_qubits

    @property
    def effective_alpha(self) -> float:
        """Returns the effective alpha coefficient taking into account the quadratic nature of the signal."""
        return self.alpha * self.axis.period**self.power

    @property
    def encoding(self) -> EncodingType:
        """Returns the encoding type of the quantum signal."""
        return self.axis.encoding


class QuadraticQuantumSignal(PolynomialQuantumSignal):
    """A quadratic quantum signal. Also called an intensity signal.

    Assumes the signal has the form f(x) = alpha * x^2 where x are the axis values.
    """

    axis: GenericAxis
    alpha: float

    power: int = 2
    """The power of the polynomial quantum signal, fixed to 2."""

    def __init__(self, axis: GenericAxis, alpha: float) -> None:
        """Initializes the QuadraticQuantumSignal."""
        self.axis = axis
        self.alpha = alpha


# class GenericQuantumSignal:
#     """A generic quantum signal having no specific structure imposed on it. It is used to be exported to other signal types such as a LinearSignal, QuadraticSignal, etc."""

#     _data: npt.NDArray[np.float64]
#     """The data of the quantum signal."""

#     def __init__(self, data: npt.ArrayLike) -> None:
#         """Initializes the GenericQuantumSignal with the given data."""
#         self._data = np.asarray(data, dtype=np.float64)

#     def to_quadratic(self) -> "QuadraticQuantumSignal":
#         """Converts the generic quantum signal to a quadratic quantum signal."""
#         return QuadraticQuantumSignal.from_data(self._data)

#     def __mul__(self, other: int | float) -> "GenericQuantumSignal":
#         """Defines the multiplication of the GenericQuantumSignal with a scalar."""
#         if isinstance(other, int | float):  # type: ignore
#             new_data = self._data * other
#             return GenericQuantumSignal(new_data)
#         else:
#             raise ValueError("Multiplication is only defined for scalars.")

#     def __add__(self, other: "GenericQuantumSignal") -> "GenericQuantumSignal":
#         """Defines the addition of two GenericQuantumSignal objects."""
#         if isinstance(other, GenericQuantumSignal):
#             if self._data.shape != other._data.shape:
#                 raise ValueError("Signals must have the same shape to be added.")
#             new_data = self._data + other._data
#             return GenericQuantumSignal(new_data)
#         else:
#             raise ValueError(
#                 "Addition is only defined between GenericQuantumSignal objects."
#             )


# class QuadraticQuantumSignal:
#     """A quadratic quantum signal. Also called an intensity signal."""

#     alpha: float
#     """The alpha parameter of the quadratic quantum signal."""
#     statevector: Statevector
#     """Returns the statevector representation of the quantum signal."""

#     def __init__(self, alpha: float, statevector: Statevector) -> None:
#         """Initializes the QuadraticQuantumSignal."""
#         self.alpha = alpha
#         self.statevector = statevector

#     @classmethod
#     def from_data(cls, data: npt.ArrayLike) -> "QuadraticQuantumSignal":
#         """Creates a QuadraticQuantumSignal from raw data."""
#         alpha, state = extract_alpha_and_state_from_generic_signal(
#             np.asarray(data), power=2
#         )
#         return cls(alpha, state)

#     @property
#     def num_qubits(self) -> int:
#         """Returns the number of qubits required to represent the statevector."""
#         return 0 if self.statevector.num_qubits is None else self.statevector.num_qubits
