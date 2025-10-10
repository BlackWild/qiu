"""Helper types for the qiskit_signals package."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

SignalFunctionType = Callable[[np.ndarray], np.ndarray]
RealSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.float64]]
ComplexSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.complex128]]

EncodingType = Literal["twos_complement", "unsigned"]
