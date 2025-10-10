"""Helper types for the qiskit_signals package."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from qiskit_signals.quantum_axis import GenericAxis

SignalFunctionType = Callable[[np.ndarray], np.ndarray]
RealSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.float64]]
ComplexSignalFunctionType = Callable[[np.ndarray], npt.NDArray[np.complex128]]

AxisFunctionType = Callable[[GenericAxis], np.ndarray]

EncodingType = Literal["twos_complement", "unsigned"]
