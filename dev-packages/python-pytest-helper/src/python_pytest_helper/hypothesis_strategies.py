"""Hypothesis strategies for numbers, axes and signals, shared across the monorepo.

Numbers are bounded by `MAX_MAGNITUDE` by default, far from overflowing, but include
zero and subnormal numbers, whose loss of precision the comparisons of
`python_pytest_helper.assertions` account for.

The strategies compose: axis strategies take strategies of their sizes, spacings and
orderings, and signal strategies take a strategy of their axes, e.g.
`monomial_signals(position_axes(orderings=st.just(IndexOrdering.FFT)))`.
"""

from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from python_signals.algebraic_signal import AlgebraicSignal, PolynomialSignal
from python_signals.integer_axis import IndexOrdering, IntegerAxis
from python_signals.physical_axis import (
    AngularWavenumberAxis,
    AxisDomain,
    MomentumAxis,
    PhysicalAxis,
    PositionAxis,
    SpatialFrequencyAxis,
)
from python_signals.signal import Signal

MAX_MAGNITUDE = 1e3
"""The default bound of the magnitude of generated numbers.

Far enough from the largest float, about `1e308`, that sums, products and powers up
to about 100 of such numbers stay finite, so tests never see overflows by accident.
"""


def reals(max_magnitude: float = MAX_MAGNITUDE) -> st.SearchStrategy[float]:
    """A strategy for finite real numbers of bounded magnitude."""
    return st.floats(min_value=-max_magnitude, max_value=max_magnitude)


def complexes(max_magnitude: float = MAX_MAGNITUDE) -> st.SearchStrategy[complex]:
    """A strategy for finite complex numbers of bounded magnitude."""
    return st.complex_numbers(
        max_magnitude=max_magnitude, allow_nan=False, allow_infinity=False
    )


def sample_arrays(
    size: int, dtype: npt.DTypeLike = np.float64, max_magnitude: float = MAX_MAGNITUDE
) -> st.SearchStrategy[npt.NDArray]:
    """A strategy for 1D arrays of finite real or complex numbers."""
    elements = (
        complexes(max_magnitude)
        if np.issubdtype(dtype, np.complexfloating)
        else reals(max_magnitude)
    )
    return arrays(dtype, size, elements=elements)


index_orderings = st.sampled_from(list(IndexOrdering))
"""A strategy for all index orderings."""

axis_domains = st.sampled_from(list(AxisDomain))
"""A strategy for all axis domains."""


def axis_sizes(min_size: int = 1, max_size: int = 64) -> st.SearchStrategy[int]:
    """A strategy for numbers of samples of an axis."""
    return st.integers(min_value=min_size, max_value=max_size)


def power_of_two_sizes(
    min_exponent: int = 0, max_exponent: int = 6
) -> st.SearchStrategy[int]:
    """A strategy for numbers of samples `2**n`, e.g. for FFTs or `n` qubits."""
    return st.integers(min_value=min_exponent, max_value=max_exponent).map(
        lambda exponent: 2**exponent
    )


def axis_spacings(
    min_value: float = 1e-3, max_value: float = 10.0
) -> st.SearchStrategy[float]:
    """A strategy for positive sampling periods of position axes."""
    return st.floats(min_value=min_value, max_value=max_value)


# the defaults of the axis and signal strategies below
_SIZES = axis_sizes()
_SPACINGS = axis_spacings()
_ALPHAS = reals(10.0)
_POWERS = st.integers(min_value=0, max_value=4)
_COEFFICIENTS = st.floats(min_value=0.01, max_value=1.0)


@st.composite
def integer_axes(
    draw: st.DrawFn,
    sizes: st.SearchStrategy[int] = _SIZES,
    orderings: st.SearchStrategy[IndexOrdering] = index_orderings,
) -> IntegerAxis:
    """A strategy for integer axes."""
    return IntegerAxis(size=draw(sizes), ordering=draw(orderings))


@st.composite
def position_axes(
    draw: st.DrawFn,
    sizes: st.SearchStrategy[int] = _SIZES,
    spacings: st.SearchStrategy[float] = _SPACINGS,
    orderings: st.SearchStrategy[IndexOrdering] = index_orderings,
) -> PositionAxis:
    """A strategy for position axes."""
    return PositionAxis(
        size=draw(sizes), delta_x=draw(spacings), ordering=draw(orderings)
    )


_CONJUGATE_AXES: dict[AxisDomain, Callable[[PositionAxis, float], PhysicalAxis]] = {
    AxisDomain.MOMENTUM: lambda x_axis, hbar: MomentumAxis.from_position_axis(
        x_axis, hbar=hbar, keep_ordering=True
    ),
    AxisDomain.ANGULAR_WAVENUMBER: lambda x_axis, _: (
        AngularWavenumberAxis.from_position_axis(x_axis, keep_ordering=True)
    ),
    AxisDomain.SPATIAL_FREQUENCY: lambda x_axis, _: (
        SpatialFrequencyAxis.from_position_axis(x_axis, keep_ordering=True)
    ),
}


def physical_axes(
    domain: AxisDomain = AxisDomain.POSITION,
    sizes: st.SearchStrategy[int] = _SIZES,
    spacings: st.SearchStrategy[float] = _SPACINGS,
    orderings: st.SearchStrategy[IndexOrdering] = index_orderings,
    hbar: float = 1.0,
) -> st.SearchStrategy[PhysicalAxis]:
    """A strategy for axes of a domain, Fourier ones conjugate to a position axis.

    Args:
        domain: The domain of the axes.
        sizes: The numbers of samples.
        spacings: The sampling periods of the position axes, or of the position
            axes the Fourier ones are conjugate to.
        orderings: The index orderings, kept by the Fourier axes.
        hbar: The reduced Planck constant of momentum axes.
    """
    x_axes = position_axes(sizes=sizes, spacings=spacings, orderings=orderings)
    if domain == AxisDomain.POSITION:
        return x_axes
    conjugate = _CONJUGATE_AXES[AxisDomain(domain)]
    return x_axes.map(lambda x_axis: conjugate(x_axis, hbar))


_POSITION_AXES = physical_axes()


@st.composite
def sampled_signals(
    draw: st.DrawFn,
    axes: st.SearchStrategy[PhysicalAxis] = _POSITION_AXES,
    dtype: npt.DTypeLike = np.float64,
    max_magnitude: float = MAX_MAGNITUDE,
) -> Signal:
    """A strategy for signals of finite sampled values of bounded magnitude."""
    axis = draw(axes)
    return Signal(axis, draw(sample_arrays(axis.size, dtype, max_magnitude)))


@st.composite
def monomial_signals(
    draw: st.DrawFn,
    axes: st.SearchStrategy[PhysicalAxis] = _POSITION_AXES,
    alphas: st.SearchStrategy[float] = _ALPHAS,
    powers: st.SearchStrategy[int] = _POWERS,
) -> PolynomialSignal:
    """A strategy for monomial signals `alpha * x**power`."""
    return PolynomialSignal(axis=draw(axes), alpha=draw(alphas), power=draw(powers))


@st.composite
def positive_polynomial_signals(
    draw: st.DrawFn,
    axes: st.SearchStrategy[PhysicalAxis] = _POSITION_AXES,
    max_degree: int = 5,
    coefficients: st.SearchStrategy[float] = _COEFFICIENTS,
    total: float = 1.0,
) -> AlgebraicSignal:
    """A strategy for non-negative signals, absolute values of polynomials.

    The signals are scaled such that their samples sum up to `total`.
    """
    axis = draw(axes)
    degree = draw(st.integers(min_value=1, max_value=max_degree))
    polynomial = np.polynomial.Polynomial(
        draw(arrays(np.float64, degree + 1, elements=coefficients))
    )
    scale = total / np.sum(np.abs(polynomial(axis.values)))
    return AlgebraicSignal(axis, lambda x: scale * np.abs(polynomial(x)))
