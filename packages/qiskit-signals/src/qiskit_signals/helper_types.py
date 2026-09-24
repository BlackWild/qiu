"""Helper types for the qiskit_signals package."""

from python_encore.enum import ExtendedEnum
from python_signals.algebraic_signal import (
    SignalFunctionType,
)
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain

__all__ = [
    "AxisType",
    "EncodingType",
    "SignalFunctionType",
]

AxisType = AxisDomain
"""Types of quantum axes, i.e. the physical domain the axis lives in."""


class EncodingType(ExtendedEnum):
    """Encoding types for the quantum signals.

    The encoding fixes which integer index each computational basis state
    represents, and thus the index ordering of the underlying axis.
    """

    TWOS_COMPLEMENT = "twos_complement"
    UNSIGNED = "unsigned"
    TWOS_COMPLEMENT_MIRRORED = "twos_complement_mirrored"

    @property
    def index_ordering(self) -> IndexOrdering:
        """Return the index ordering of an axis with this encoding."""
        if self == EncodingType.UNSIGNED:
            return IndexOrdering.NATURAL
        if self == EncodingType.TWOS_COMPLEMENT:
            return IndexOrdering.FFT
        if self == EncodingType.TWOS_COMPLEMENT_MIRRORED:
            return IndexOrdering.CENTERED
        raise ValueError(f"Unknown encoding type: {self}")
