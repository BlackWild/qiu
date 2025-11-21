"""Helper types for the qiskit_signals package."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from python_encore.enum import ExtendedEnum

SignalFunctionType = Callable[[np.ndarray], np.ndarray]
RealSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.float64]]
ComplexSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.complex128]]


class EncodingType(ExtendedEnum):
    """Encoding types for the quantum signals."""

    TWOS_COMPLEMENT = "twos_complement"
    UNSIGNED = "unsigned"
    TWOS_COMPLEMENT_MIRRORED = "twos_complement_mirrored"


class AxisType(ExtendedEnum):
    """Types of quantum axes.

    Tracks which domain the axis lives in. Options are the position domain or its conjugate momentum domain.
    """

    POSITION = "position"
    MOMENTUM = "momentum"
    ANGULAR_WAVENUMBER = "angular_wavenumber"
    SPATIAL_FREQUENCY = "spatial_frequency"

    @property
    def is_in_fourier_domain(self) -> bool:
        if (
            self == AxisType.MOMENTUM
            or self == AxisType.ANGULAR_WAVENUMBER
            or self == AxisType.SPATIAL_FREQUENCY
        ):
            return True
        if self == AxisType.POSITION:
            return False
        raise ValueError("Undefined axis domain")
