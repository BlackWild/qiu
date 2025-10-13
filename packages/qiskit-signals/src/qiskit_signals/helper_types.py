"""Helper types for the qiskit_signals package."""

from collections.abc import Callable
from enum import Enum

import numpy as np
import numpy.typing as npt

SignalFunctionType = Callable[[np.ndarray], np.ndarray]
RealSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.float64]]
ComplexSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.complex128]]


class EncodingType(Enum):
    """Encoding types for the quantum signals."""

    TWOS_COMPLEMENT = "twos_complement"
    UNSIGNED = "unsigned"
