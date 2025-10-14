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


class AxisType(ExtendedEnum):
    """Types of quantum axes."""

    POSITION = "position"
    MOMENTUM = "momentum"
