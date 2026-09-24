"""Collection of tools to represent quantum signals.

Quantum signals are the algebraic signals of `python_signals` on quantum axes, which
additionally know the number of qubits and the encoding they are represented with.
"""

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import (
    AlgebraicSignal,
    PolynomialSignal,
    QuadraticSignal,
    SignalFunctionType,
)

from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import GenericAxis

# - MARK: Generic


class GenericQuantumSignal(AlgebraicSignal):
    """A generic quantum signal having no specific structure imposed on it."""

    axis: GenericAxis

    def __init__(self, axis: GenericAxis, signal_function: SignalFunctionType) -> None:
        """Initializes the GenericQuantumSignal with the given axis and signal function."""
        super().__init__(axis=axis, function=signal_function)

    @property
    def signal_function(self) -> SignalFunctionType:
        """Returns the function mapping the axis values to the signal values."""
        return self.function

    @property
    def normalized_data(self) -> npt.NDArray[np.number]:
        """Returns the data of the quantum signal normalized to unit Euclidean norm."""
        return self.to_signal().normalized_data

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the signal."""
        return self.axis.num_qubits

    @property
    def dimension(self) -> int:
        """Returns the dimension of the signal."""
        return self.axis.dimension


class PolynomialQuantumSignal(PolynomialSignal):
    """A polynomial quantum signal."""

    axis: GenericAxis

    def to_generic(self) -> GenericQuantumSignal:
        """Converts the polynomial quantum signal to a generic quantum signal."""
        alpha, power = self.alpha, self.power
        return GenericQuantumSignal(
            axis=self.axis, signal_function=lambda x: alpha * x**power
        )

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the signal."""
        return self.axis.num_qubits

    @property
    def encoding(self) -> EncodingType:
        """Returns the encoding type of the quantum signal."""
        return self.axis.encoding


class QuadraticQuantumSignal(PolynomialQuantumSignal, QuadraticSignal):
    """A quadratic quantum signal. Also called an intensity signal.

    Assumes the signal has the form f(x) = alpha * x^2 where x are the axis values.
    """
