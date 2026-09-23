"""Uniformly sampled axes of integer indices."""

import numpy as np
import numpy.typing as npt
from python_encore.enum import ExtendedEnum


class IndexOrdering(ExtendedEnum):
    """Orderings of the integer indices of the samples of an axis.

    For an axis of `N` samples, the sample at array position `k` gets the index:

    - `NATURAL`: `k`, i.e. `0, 1, ..., N-1`.
    - `FFT`: the ordering of `numpy.fft.fftfreq`, i.e. the non-negative indices
      `0, 1, ..., ceil(N/2)-1` followed by the negative ones `-floor(N/2), ..., -1`.
    - `CENTERED`: the `FFT` ordering after `numpy.fft.fftshift`, i.e. the ascending
      indices `-floor(N/2), ..., ceil(N/2)-1`.
    """

    NATURAL = "natural"
    FFT = "fft"
    CENTERED = "centered"


class IntegerAxis:
    """A uniformly sampled axis of integer indices.

    The sample at array position `k` has the integer index `index[k]`, given by the
    ordering of the axis.
    """

    size: int
    """Number of samples."""
    ordering: IndexOrdering
    """Ordering of the integer indices of the samples."""

    def __init__(self, size: int, ordering: IndexOrdering) -> None:
        """Initialize an integer axis.

        Args:
            size: Number of samples, at least 1.
            ordering: Ordering of the integer indices of the samples.
        """
        if size < 1:
            raise ValueError(f"The size must be at least 1, got {size}.")

        self.size = size
        self.ordering = IndexOrdering(ordering)

    def __repr__(self) -> str:
        """Return a readable representation of the axis."""
        return (
            f"{type(self).__name__}(size={self.size}, ordering={self.ordering.value})"
        )

    @property
    def index(self) -> npt.NDArray[np.int_]:
        """Return the integer indices of the samples according to the ordering."""
        k = np.arange(self.size)
        if self.ordering == IndexOrdering.NATURAL:
            return k
        if self.ordering == IndexOrdering.FFT:
            return np.where(k < (self.size + 1) // 2, k, k - self.size)
        if self.ordering == IndexOrdering.CENTERED:
            return k - self.size // 2
        raise ValueError(f"Unknown index ordering: {self.ordering}")
