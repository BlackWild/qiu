"""Unit tests for hypothesis_strategies.py."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import RTOL, assert_close
from python_pytest_helper.hypothesis_strategies import (
    MAX_MAGNITUDE,
    axis_sizes,
    integer_axes,
    monomial_signals,
    physical_axes,
    position_axes,
    positive_polynomial_signals,
    power_of_two_sizes,
    sample_arrays,
    sampled_signals,
)
from qiu_signals.algebraic_signal import AlgebraicSignal, PolynomialSignal
from qiu_signals.integer_axis import IndexOrdering, IntegerAxis
from qiu_signals.physical_axis import AxisDomain, PhysicalAxis, PositionAxis
from qiu_signals.signal import Signal


def within_max_magnitude(values: npt.NDArray) -> bool:
    """Whether the magnitudes are bounded by MAX_MAGNITUDE, up to their rounding."""
    return bool(np.all(np.abs(values) <= MAX_MAGNITUDE * (1 + RTOL)))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@given(data=st.data())
def test_sample_arrays(dtype: npt.DTypeLike, data: st.DataObject):
    """Test that the arrays are finite, bounded and of the given type and size."""
    array = data.draw(sample_arrays(5, dtype))
    assert array.shape == (5,)
    assert array.dtype == dtype
    assert np.all(np.isfinite(array))
    assert within_max_magnitude(array)


@given(size=power_of_two_sizes(min_exponent=1, max_exponent=4))
def test_power_of_two_sizes(size: int):
    """Test that the sizes are the powers of 2 within the exponents."""
    assert size in {2, 4, 8, 16}


@given(axis=integer_axes(sizes=axis_sizes(max_size=4)))
def test_integer_axes(axis: IntegerAxis):
    """Test that the integer axes have the given sizes."""
    assert isinstance(axis, IntegerAxis)
    assert 1 <= axis.size <= 4


@given(axis=position_axes(orderings=st.just(IndexOrdering.FFT)))
def test_position_axes(axis: PositionAxis):
    """Test that the position axes have a positive period and the given ordering."""
    assert isinstance(axis, PositionAxis)
    assert axis.period > 0
    assert axis.ordering is IndexOrdering.FFT


@pytest.mark.parametrize("domain", list(AxisDomain))
@given(data=st.data())
def test_physical_axes(domain: AxisDomain, data: st.DataObject):
    """Test that the axes live in the domain and keep the given ordering."""
    axis = data.draw(physical_axes(domain, orderings=st.just(IndexOrdering.CENTERED)))
    assert isinstance(axis, PhysicalAxis)
    assert axis.domain is domain
    assert axis.ordering is IndexOrdering.CENTERED


@given(signal=sampled_signals(dtype=np.complex128))
def test_sampled_signals(signal: Signal):
    """Test that the sampled values are finite and bounded."""
    assert isinstance(signal, Signal)
    assert np.iscomplexobj(signal.data)
    assert within_max_magnitude(signal.data)


@given(signal=monomial_signals(powers=st.just(3)))
def test_monomial_signals(signal: PolynomialSignal):
    """Test that the monomials have the given power."""
    assert isinstance(signal, PolynomialSignal)
    assert signal.power == 3
    assert_close(signal.data, signal.alpha * signal.axis.values**3)


@given(signal=positive_polynomial_signals(total=0.5))
def test_positive_polynomial_signals(signal: AlgebraicSignal):
    """Test that the signals are non-negative and sum up to the total."""
    assert isinstance(signal, AlgebraicSignal)
    assert np.all(np.isfinite(signal.data))
    assert np.all(signal.data >= 0)
    assert_close(np.sum(signal.data), 0.5)
