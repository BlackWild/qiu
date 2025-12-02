"""Module defining quantum axes for position, momentum, and angular frequency."""

from abc import ABC

import numpy as np
import numpy.typing as npt

from qiskit_signals.helper_types import AxisType, EncodingType


class GenericAxis(ABC):
    """A generic quantum axis class."""

    encoding: EncodingType
    period: float
    num_qubits: int
    axis_type: AxisType

    def __init__(
        self,
        num_qubits: int,
        period: float,
        encoding: EncodingType,
        axis_type: AxisType,
    ):
        self.encoding = encoding
        self.period = period
        self.num_qubits = num_qubits
        self.axis_type = axis_type

    @property
    def index(self) -> npt.NDArray:
        """Return the integer index values corresponding to the encoding."""
        if self.encoding == EncodingType.UNSIGNED:
            return np.arange(2**self.num_qubits)
        elif self.encoding == EncodingType.TWOS_COMPLEMENT:
            half = 2 ** (self.num_qubits - 1)
            return np.concatenate((np.arange(0, half), np.arange(-half, 0)))
        elif self.encoding == EncodingType.TWOS_COMPLEMENT_MIRRORED:
            half = 2 ** (self.num_qubits - 1)
            return np.concatenate((np.arange(-half, 0), np.arange(0, half)))
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

    @property
    def is_fourier_domain_axis(self) -> bool:
        return self.axis_type.is_in_fourier_domain


class PositionAxis(GenericAxis):
    """A quantum axis representing position."""

    def __init__(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Initialize a position axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values.
            encoding: Encoding type, either 'twos_complement' or 'unsigned'.
        """
        super().__init__(
            num_qubits=num_qubits,
            period=delta_x,
            encoding=encoding,
            axis_type=AxisType.POSITION,
        )


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
            hbar: Reduced Planck's constant.
        """
        period = 2 * np.pi * hbar / (2**num_qubits * delta_x)
        super().__init__(
            num_qubits=num_qubits,
            period=period,
            encoding=encoding,
            axis_type=AxisType.MOMENTUM,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        hbar: float,
        keep_encoding: bool = False,
    ) -> "MomentumAxis":
        """Create a MomentumAxis from a given PositionAxis.

        Args:
            position_axis: An instance of PositionAxis.
            hbar: Reduced Planck's constant (default is 1.0 for natural units).
            keep_encoding: If True, retain the encoding of the position axis; otherwise, use 'twos_complement'.

        Returns:
            An instance of MomentumAxis.
        """
        return cls(
            num_qubits=position_axis.num_qubits,
            delta_x=position_axis.period,
            encoding=position_axis.encoding
            if keep_encoding
            else EncodingType.TWOS_COMPLEMENT,
            hbar=hbar,
        )


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
        super().__init__(
            num_qubits=num_qubits,
            period=period,
            encoding=encoding,
            axis_type=AxisType.ANGULAR_WAVENUMBER,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        keep_encoding: bool = False,
    ) -> "AngularWavenumberAxis":
        return cls(
            num_qubits=position_axis.num_qubits,
            delta_x=position_axis.period,
            encoding=position_axis.encoding
            if keep_encoding
            else EncodingType.TWOS_COMPLEMENT,
        )


class SpatialFrequencyAxis(GenericAxis):
    """A quantum axis representing spatial frequency."""

    def __init__(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Initialize a spatial frequency axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values (used to compute momentum spacing).
            encoding: Encoding type, either 'twos_complement' or 'unsigned'.
        """
        period = 1 / (2**num_qubits * delta_x)
        super().__init__(
            num_qubits=num_qubits,
            period=period,
            encoding=encoding,
            axis_type=AxisType.SPATIAL_FREQUENCY,
        )
