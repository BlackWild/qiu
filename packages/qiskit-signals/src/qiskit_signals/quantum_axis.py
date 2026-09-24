"""Module defining quantum axes for position, momentum, and angular frequency.

A quantum axis is an axis of `2**num_qubits` samples, whose index ordering is
given by the encoding of the integers in the computational basis states.
"""

import numpy as np
import numpy.typing as npt
from python_signals.physical_axis import PhysicalAxis, reciprocal_period

from qiskit_signals.helper_types import AxisType, EncodingType


class GenericAxis(PhysicalAxis):
    """A generic quantum axis class."""

    encoding: EncodingType
    num_qubits: int

    def __init__(
        self,
        num_qubits: int,
        period: float,
        encoding: EncodingType,
        axis_type: AxisType,
    ):
        """Initialize a quantum axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            period: Spacing between neighboring axis values.
            encoding: Encoding of the integer indices in the basis states.
            axis_type: Physical domain the axis lives in.
        """
        self.num_qubits = num_qubits
        self.encoding = EncodingType(encoding)
        super().__init__(
            size=2**num_qubits,
            period=period,
            ordering=self.encoding.index_ordering,
            domain=axis_type,
        )

    @property
    def axis_type(self) -> AxisType:
        """Return the physical domain the axis lives in."""
        return self.domain

    @property
    def axis_values(self) -> npt.NDArray[np.float64]:
        """Return the physical axis values corresponding to the encoding and period."""
        return self.values

    @property
    def dimension(self) -> int:
        """Return the dimension of the axis (number of discrete values)."""
        return self.size

    @property
    def is_fourier_domain_axis(self) -> bool:
        """Whether the axis lives in a Fourier conjugate domain of position."""
        return self.is_fourier_domain


class PositionAxis(GenericAxis):
    """A quantum axis representing position."""

    def __init__(self, num_qubits: int, delta_x: float, encoding: EncodingType):
        """Initialize a position axis.

        Args:
            num_qubits: Number of qubits representing the axis.
            delta_x: Spacing between discrete position values.
            encoding: Encoding type of the integer indices.
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
            encoding: Encoding type of the integer indices.
            hbar: Reduced Planck's constant.
        """
        super().__init__(
            num_qubits=num_qubits,
            period=reciprocal_period(
                2**num_qubits, delta_x, AxisType.MOMENTUM, hbar=hbar
            ),
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
            hbar: Reduced Planck's constant.
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
            encoding: Encoding type of the integer indices.
        """
        super().__init__(
            num_qubits=num_qubits,
            period=reciprocal_period(
                2**num_qubits, delta_x, AxisType.ANGULAR_WAVENUMBER
            ),
            encoding=encoding,
            axis_type=AxisType.ANGULAR_WAVENUMBER,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        keep_encoding: bool = False,
    ) -> "AngularWavenumberAxis":
        """Create an AngularWavenumberAxis from a given PositionAxis.

        Args:
            position_axis: An instance of PositionAxis.
            keep_encoding: If True, retain the encoding of the position axis; otherwise, use 'twos_complement'.

        Returns:
            An instance of AngularWavenumberAxis.
        """
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
            encoding: Encoding type of the integer indices.
        """
        super().__init__(
            num_qubits=num_qubits,
            period=reciprocal_period(
                2**num_qubits, delta_x, AxisType.SPATIAL_FREQUENCY
            ),
            encoding=encoding,
            axis_type=AxisType.SPATIAL_FREQUENCY,
        )
