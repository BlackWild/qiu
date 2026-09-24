"""Signals given by their samples on a physical axis."""

import numpy as np
import numpy.typing as npt

from python_signals.physical_axis import PhysicalAxis


class Signal:
    """A signal given by its sampled values on a physical axis.

    The sample `data[k]` is the value of the signal at `axis.values[k]`.
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
