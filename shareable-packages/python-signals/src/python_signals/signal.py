"""Signals given by their samples on a physical axis."""

from typing import Any

import numpy as np
import numpy.typing as npt

from python_signals.arithmetic import (
    ArithmeticOperators,
    BinaryOperator,
    is_scalar,
    require_same_axis,
)
from python_signals.physical_axis import PhysicalAxis


class Signal(ArithmeticOperators):
    """A signal given by its sampled values on a physical axis.

    The sample `data[k]` is the value of the signal at `axis.values[k]`.

    Signals support the arithmetic operators `+`, `-`, `*`, `/` and `**`, elementwise
    with scalars and with signals on an equal axis, e.g. `2 * signal + other`. The
    result is a new `Signal`.
    """

    axis: PhysicalAxis
    """The axis the signal is sampled on."""
    data: npt.NDArray[np.number]
    """The sampled values of the signal, real or complex, one per axis sample."""

    def __init__(self, axis: PhysicalAxis, data: npt.ArrayLike) -> None:
        """Initialize the signal from its sampled values.

        Args:
            axis: The axis the signal is sampled on.
            data: The sampled values, a one-dimensional numeric array with one value
                per axis sample.
        """
        data = np.asarray(data)
        if data.shape != (axis.size,):
            raise ValueError(
                f"The data must have the shape ({axis.size},) of the axis, "
                f"got {data.shape}."
            )
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"The data must be numeric, got dtype {data.dtype}.")

        self.axis = axis
        self.data = data

    def __repr__(self) -> str:
        """Return a readable representation of the signal."""
        return f"{type(self).__name__}(axis={self.axis!r}, dtype={self.data.dtype})"

    @property
    def size(self) -> int:
        """Return the number of samples of the signal."""
        return self.axis.size

    @property
    def normalized_data(self) -> npt.NDArray[np.number]:
        """Return the sampled values normalized to unit Euclidean norm.

        An all-zero signal is returned unchanged.
        """
        norm = np.linalg.norm(self.data)
        if norm == 0:
            return self.data
        return self.data / norm

    def _binary(self, other: Any, op: BinaryOperator, reflected: bool) -> Any:
        """Combine elementwise with a scalar or a signal on an equal axis."""
        if is_scalar(other):
            other_data = other
        elif isinstance(other, Signal):
            require_same_axis(self.axis, other.axis)
            other_data = other.data
        else:
            return NotImplemented

        if reflected:
            return Signal(self.axis, op(other_data, self.data))
        return Signal(self.axis, op(self.data, other_data))
