"""Helper types."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

RealSignalType = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
ComplexSignalType = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complexfloating]]
