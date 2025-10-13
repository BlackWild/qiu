"""Module defining quantum axes for position, momentum, and angular frequency."""

from abc import ABC

import numpy as np
import numpy.typing as npt

from qiskit_signals.helper_types import EncodingType


class GenericAxis(ABC):
    """A generic quantum axis class."""

    encoding: EncodingType
    period: float
    num_qubits: int

    def __init__(self, num_qubits: int, period: float, encoding: EncodingType):
        self.encoding = encoding
        self.period = period
        self.num_qubits = num_qubits

    @property
    def index(self) -> npt.NDArray:
        """Return the integer index values corresponding to the encoding."""
        if self.encoding == "unsigned":
            return np.arange(2**self.num_qubits)
        elif self.encoding == "twos_complement":
            half = 2 ** (self.num_qubits - 1)
            return np.concatenate((np.arange(0, half), np.arange(-half, 0)))
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding}")

    @property
    def axis_values(self) -> npt.NDArray[np.float64]:
        """Return the physical axis values corresponding to the encoding and period."""
        return self.index * self.period

    @property
    def dimension(self) -> int:
        """Return the dimension of the axis (number of discrete values)."""
        return 2**self.num_qubits

    @property
    def sampling_window_length(self) -> float:
        """Return the total length of the sampling window."""
        return self.dimension * self.period


class PositionAxis(GenericAxis):
    """A quantum axis representing position."""

    def __init__(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Initialize a position axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values.
            encoding: Encoding type, either 'twos_complement' or 'unsigned'.
        """
        super().__init__(num_qubits=num_qubits, period=delta_x, encoding=encoding)


class MomentumAxis(GenericAxis):
    """A quantum axis representing momentum."""

    def __init__(
        self, num_qubits: int, delta_x: float, encoding: EncodingType, hbar: float
    ):
        """Initialize a momentum axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values (used to compute momentum spacing).
            encoding: Encoding type, either 'twos_complement' or 'unsigned'.
        """
        period = 2 * np.pi * hbar / (2**num_qubits * delta_x)
        super().__init__(num_qubits=num_qubits, period=period, encoding=encoding)


class AngularWavenumberAxis(GenericAxis):
    """A quantum axis representing angular wavenumber."""

    def __init__(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Initialize an angular wavenumber axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values (used to compute momentum spacing).
            encoding: Encoding type, either 'twos_complement' or 'unsigned'.
        """
        period = 2 * np.pi / (2**num_qubits * delta_x)
        super().__init__(num_qubits=num_qubits, period=period, encoding=encoding)
