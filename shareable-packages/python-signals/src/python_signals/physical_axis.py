"""Uniformly sampled axes for position and its Fourier conjugate domains."""

import numpy as np
import numpy.typing as npt
from python_encore.enum import ExtendedEnum

from python_signals.integer_axis import IndexOrdering, IntegerAxis


class AxisDomain(ExtendedEnum):
    """Physical domains an axis can live in.

    Either the position domain, or one of its Fourier conjugate domains.
    """

    POSITION = "position"
    MOMENTUM = "momentum"
    ANGULAR_WAVENUMBER = "angular_wavenumber"
    SPATIAL_FREQUENCY = "spatial_frequency"

    @property
    def is_in_fourier_domain(self) -> bool:
        """Whether the domain is a Fourier conjugate of the position domain."""
        if (
            self == AxisDomain.MOMENTUM
            or self == AxisDomain.ANGULAR_WAVENUMBER
            or self == AxisDomain.SPATIAL_FREQUENCY
        ):
            return True
        if self == AxisDomain.POSITION:
            return False
        raise ValueError("Undefined axis domain")


def reciprocal_period(
    size: int, delta_x: float, domain: AxisDomain, *, hbar: float | None = None
) -> float:
    """Return the sampling period of the Fourier conjugate of a position axis.

    For a position axis of `size` samples spaced by `delta_x`, the discrete Fourier
    transform samples the conjugate domain with the period
        - $2 pi hbar / (N delta_x)$ for momentum,
        - $2 pi / (N delta_x)$ for angular wavenumber,
        - $1 / (N delta_x)$ for spatial frequency,
    where $N$ is the number of samples.

    Args:
        size: Number of samples of the position axis.
        delta_x: Spacing between the samples of the position axis.
        domain: The Fourier conjugate domain to compute the period for.
        hbar: Reduced Planck's constant, in the units of choice. Must be given for
            the momentum domain, and only for it.

    Returns:
        The spacing between the samples of the conjugate axis.
    """

    if (domain == AxisDomain.MOMENTUM) != (hbar is not None):
        raise ValueError(
            "hbar must be given for the momentum domain, and only for it, "
            f"got hbar={hbar} for the {AxisDomain(domain).value} domain."
        )

    window_length = size * delta_x
    if domain == AxisDomain.MOMENTUM:
        return 2 * np.pi * hbar / window_length  # type: ignore[operator]
    if domain == AxisDomain.ANGULAR_WAVENUMBER:
        return 2 * np.pi / window_length
    if domain == AxisDomain.SPATIAL_FREQUENCY:
        return 1 / window_length
    raise ValueError(f"Not a Fourier conjugate domain: {domain}")


class PhysicalAxis(IntegerAxis):
    """A uniformly sampled axis in a physical domain.

    The sample at array position `k` lies at `index[k] * period`, where the integer
    indices are given by the ordering of the axis.
    """

    period: float
    """Spacing between neighboring samples."""
    domain: AxisDomain
    """Physical domain the axis lives in."""

    def __init__(
        self,
        size: int,
        period: float,
        ordering: IndexOrdering,
        domain: AxisDomain,
    ) -> None:
        """Initialize a physical axis.

        Args:
            size: Number of samples, at least 1.
            period: Spacing between neighboring samples, positive and finite.
            ordering: Ordering of the integer indices of the samples.
            domain: Physical domain the axis lives in.
        """
        super().__init__(size=size, ordering=ordering)

        if not (np.isfinite(period) and period > 0):
            raise ValueError(f"The period must be positive and finite, got {period}.")

        self.period = period
        self.domain = AxisDomain(domain)

    def __repr__(self) -> str:
        """Return a readable representation of the axis."""
        return (
            f"{type(self).__name__}(size={self.size}, period={self.period}, "
            f"ordering={self.ordering.value}, domain={self.domain.value})"
        )

    def _key(self) -> tuple:
        """Return the attributes defining the axis, for equality and hashing."""
        return (self.size, self.ordering, self.period, self.domain)

    @property
    def values(self) -> npt.NDArray[np.number]:
        """Return the physical values of the samples."""
        return self.index * self.period

    @property
    def sampling_window_length(self) -> float:
        """Return the total length of the sampling window."""
        return self.size * self.period

    @property
    def is_fourier_domain(self) -> bool:
        """Whether the axis lives in a Fourier conjugate domain of position."""
        return self.domain.is_in_fourier_domain


class PositionAxis(PhysicalAxis):
    """An axis representing position."""

    def __init__(self, size: int, delta_x: float, ordering: IndexOrdering) -> None:
        """Initialize a position axis.

        Args:
            size: Number of samples.
            delta_x: Spacing between the position samples.
            ordering: Ordering of the integer indices of the samples.
        """
        super().__init__(
            size=size,
            period=delta_x,
            ordering=ordering,
            domain=AxisDomain.POSITION,
        )


class MomentumAxis(PhysicalAxis):
    """An axis representing momentum, the Fourier conjugate of a position axis."""

    def __init__(self, size: int, delta_p: float, ordering: IndexOrdering) -> None:
        """Initialize a momentum axis.

        Args:
            size: Number of samples.
            delta_p: Spacing between the momentum samples.
            ordering: Ordering of the integer indices of the samples.
        """
        super().__init__(
            size=size,
            period=delta_p,
            ordering=ordering,
            domain=AxisDomain.MOMENTUM,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        hbar: float,
        keep_ordering: bool = False,
    ) -> "MomentumAxis":
        """Create the momentum axis conjugate to a position axis.

        The momentum spacing is $2 pi hbar / (N delta_x)$, see `reciprocal_period`.

        Args:
            position_axis: The position axis.
            hbar: Reduced Planck's constant, in the units of choice.
            keep_ordering: If True, retain the ordering of the position axis;
                otherwise, use the `FFT` ordering of the discrete Fourier transform.

        Returns:
            The conjugate momentum axis.
        """
        return cls(
            size=position_axis.size,
            delta_p=reciprocal_period(
                position_axis.size,
                position_axis.period,
                AxisDomain.MOMENTUM,
                hbar=hbar,
            ),
            ordering=position_axis.ordering if keep_ordering else IndexOrdering.FFT,
        )


class AngularWavenumberAxis(PhysicalAxis):
    """An axis representing angular wavenumber, conjugate of a position axis."""

    def __init__(self, size: int, delta_k: float, ordering: IndexOrdering) -> None:
        """Initialize an angular wavenumber axis.

        Args:
            size: Number of samples.
            delta_k: Spacing between the angular wavenumber samples.
            ordering: Ordering of the integer indices of the samples.
        """
        super().__init__(
            size=size,
            period=delta_k,
            ordering=ordering,
            domain=AxisDomain.ANGULAR_WAVENUMBER,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        keep_ordering: bool = False,
    ) -> "AngularWavenumberAxis":
        """Create the angular wavenumber axis conjugate to a position axis.

        The angular wavenumber spacing is $2 pi / (N delta_x)$, see
        `reciprocal_period`.

        Args:
            position_axis: The position axis.
            keep_ordering: If True, retain the ordering of the position axis;
                otherwise, use the `FFT` ordering of the discrete Fourier transform.

        Returns:
            The conjugate angular wavenumber axis.
        """
        return cls(
            size=position_axis.size,
            delta_k=reciprocal_period(
                position_axis.size,
                position_axis.period,
                AxisDomain.ANGULAR_WAVENUMBER,
            ),
            ordering=position_axis.ordering if keep_ordering else IndexOrdering.FFT,
        )


class SpatialFrequencyAxis(PhysicalAxis):
    """An axis representing spatial frequency, conjugate of a position axis."""

    def __init__(self, size: int, delta_f: float, ordering: IndexOrdering) -> None:
        """Initialize a spatial frequency axis.

        Args:
            size: Number of samples.
            delta_f: Spacing between the spatial frequency samples.
            ordering: Ordering of the integer indices of the samples.
        """
        super().__init__(
            size=size,
            period=delta_f,
            ordering=ordering,
            domain=AxisDomain.SPATIAL_FREQUENCY,
        )

    @classmethod
    def from_position_axis(
        cls,
        position_axis: PositionAxis,
        keep_ordering: bool = False,
    ) -> "SpatialFrequencyAxis":
        """Create the spatial frequency axis conjugate to a position axis.

        The spatial frequency spacing is $1 / (N delta_x)$, see `reciprocal_period`.

        Args:
            position_axis: The position axis.
            keep_ordering: If True, retain the ordering of the position axis;
                otherwise, use the `FFT` ordering of the discrete Fourier transform.

        Returns:
            The conjugate spatial frequency axis.
        """
        return cls(
            size=position_axis.size,
            delta_f=reciprocal_period(
                position_axis.size,
                position_axis.period,
                AxisDomain.SPATIAL_FREQUENCY,
            ),
            ordering=position_axis.ordering if keep_ordering else IndexOrdering.FFT,
        )
