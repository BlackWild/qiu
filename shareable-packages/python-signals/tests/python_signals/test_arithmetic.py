"""Unit tests for the arithmetic operators of the signals."""

import operator

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_signals.algebraic_signal import (
    AlgebraicSignal,
    PolynomialSignal,
    QuadraticSignal,
)
from python_signals.arithmetic import is_scalar
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal

AXIS = PositionAxis(8, 0.5, IndexOrdering.FFT)
OTHER_AXIS = PositionAxis(8, 0.25, IndexOrdering.FFT)

# operators on positive data, so that / and ** are well defined
operators = st.sampled_from(
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]
)
scalars = st.one_of(
    st.floats(min_value=0.5, max_value=2.0),
    st.integers(min_value=1, max_value=3),
    st.floats(min_value=0.5, max_value=2.0).map(np.float64),
)


def positive(x):
    """A positive test function."""
    return 1 + np.cos(x) ** 2


def other_positive(x):
    """Another positive test function."""
    return 2 + np.sin(x)


@pytest.fixture(scope="module")
def sympy():
    """The SymPy module, skipping the test if SymPy is not installed."""
    return pytest.importorskip("sympy")


class TestIsScalar:
    """Test is_scalar."""

    @pytest.mark.parametrize("value", [1, 2.5, 1j, np.float64(1), np.int32(2), True])
    def test_scalars(self, value):
        """Test that Python and NumPy numbers are scalars."""
        assert is_scalar(value)

    @pytest.mark.parametrize("value", [np.ones(3), [1, 2], "1", None])
    def test_non_scalars(self, value):
        """Test that arrays and other objects are not scalars."""
        assert not is_scalar(value)


class TestSignalArithmetic:
    """Test the operators of sampled signals."""

    @given(op=operators, scalar=scalars)
    def test_with_scalars(self, op, scalar):
        """Test elementwise operations with scalars, from both sides."""
        signal = Signal(AXIS, positive(AXIS.values))

        for result, expected in [
            (op(signal, scalar), op(signal.data, scalar)),
            (op(scalar, signal), op(scalar, signal.data)),
        ]:
            assert type(result) is Signal
            assert result.axis is AXIS
            assert_close(result.data, expected)

    @given(op=operators)
    def test_with_signals(self, op):
        """Test elementwise operations between signals on equal axes."""
        signal = Signal(AXIS, positive(AXIS.values))
        other = Signal(PositionAxis(8, 0.5, "fft"), other_positive(AXIS.values))  # type: ignore[arg-type]

        result = op(signal, other)
        assert type(result) is Signal
        assert_close(result.data, op(signal.data, other.data))

    def test_negation(self):
        """Test the unary operators."""
        signal = Signal(AXIS, positive(AXIS.values))
        np.testing.assert_array_equal((-signal).data, -signal.data)
        assert +signal is signal

    def test_complex_values(self):
        """Test that complex signals and scalars are supported."""
        signal = Signal(AXIS, np.exp(1j * AXIS.values))
        assert_close((1j * signal).data, 1j * signal.data)

    def test_axes_must_be_equal(self):
        """Test that signals on different axes cannot be combined."""
        with pytest.raises(ValueError, match="equal axes"):
            _ = Signal(AXIS, np.ones(8)) + Signal(OTHER_AXIS, np.ones(8))

    def test_arrays_are_rejected(self):
        """Test that raw arrays are not silently broadcast against the signal."""
        signal = Signal(AXIS, np.ones(8))
        with pytest.raises(TypeError):
            _ = signal * np.ones(8)
        with pytest.raises(TypeError):
            _ = np.ones(8) * signal

    def test_does_not_modify_the_operands(self):
        """Test that the operators return new signals."""
        data = np.ones(8)
        signal = Signal(AXIS, data)
        _ = signal + 1
        np.testing.assert_array_equal(signal.data, np.ones(8))


class TestAlgebraicSignalArithmetic:
    """Test the operators of algebraic signals."""

    @given(op=operators, scalar=scalars)
    def test_with_scalars(self, op, scalar):
        """Test that operations with scalars compose the functions."""
        signal = AlgebraicSignal(AXIS, positive)

        for result, expected in [
            (op(signal, scalar), op(positive(AXIS.values), scalar)),
            (op(scalar, signal), op(scalar, positive(AXIS.values))),
        ]:
            assert type(result) is AlgebraicSignal
            assert_close(result.data, expected)

    @given(op=operators)
    def test_with_algebraic_signals(self, op):
        """Test operations between algebraic signals on equal axes."""
        result = op(
            AlgebraicSignal(AXIS, positive), AlgebraicSignal(AXIS, other_positive)
        )

        assert type(result) is AlgebraicSignal
        assert_close(
            result.data, op(positive(AXIS.values), other_positive(AXIS.values))
        )
        # the result is still algebraic, i.e. evaluable off the axis
        values = np.linspace(-1, 1, 5)
        assert_close(result(values), op(positive(values), other_positive(values)))

    @given(op=operators)
    def test_with_sampled_signals(self, op):
        """Test that combining with a sampled signal samples the algebraic one."""
        algebraic = AlgebraicSignal(AXIS, positive)
        sampled = Signal(AXIS, other_positive(AXIS.values))

        for result, expected in [
            (op(algebraic, sampled), op(algebraic.data, sampled.data)),
            (op(sampled, algebraic), op(sampled.data, algebraic.data)),
        ]:
            assert type(result) is Signal
            assert_close(result.data, expected)

    def test_constant_operands(self):
        """Test that constant functions broadcast in combinations."""
        result = AlgebraicSignal(AXIS, lambda x: 2.0) * AlgebraicSignal(
            AXIS, lambda x: 3.0
        )
        np.testing.assert_array_equal(result.data, np.full(8, 6.0))

    def test_negation(self):
        """Test the unary operators."""
        signal = AlgebraicSignal(AXIS, positive)
        assert_close((-signal).data, -positive(AXIS.values))
        assert +signal is signal

    def test_axes_must_be_equal(self):
        """Test that algebraic signals on different axes cannot be combined."""
        with pytest.raises(ValueError, match="equal axes"):
            _ = AlgebraicSignal(AXIS, positive) + AlgebraicSignal(OTHER_AXIS, positive)

    def test_expressions_are_combined(self, sympy):
        """Test that the SymPy expressions of both operands are combined."""
        x = sympy.Symbol("x")
        signal = AlgebraicSignal.from_sympy(AXIS, x**2)

        assert (2 * signal + 1).expression == 2 * x**2 + 1
        assert (1 / signal).expression == x**-2
        assert (-signal).expression == -(x**2)
        assert (signal * AlgebraicSignal.from_sympy(AXIS, x + 1)).expression == x**2 * (
            x + 1
        )
        assert (2 * signal).symbol == x

    def test_expressions_in_other_symbols_are_substituted(self, sympy):
        """Test that the other operand's symbol is replaced by the signal's."""
        x, t = sympy.symbols("x t")
        result = AlgebraicSignal.from_sympy(AXIS, x**2) + AlgebraicSignal.from_sympy(
            AXIS, t
        )
        assert result.expression == x**2 + x

    def test_expressions_are_dropped_without_counterpart(self, sympy):
        """Test that combining with a function-based signal drops the expression."""
        x = sympy.Symbol("x")
        result = AlgebraicSignal.from_sympy(AXIS, x**2) + AlgebraicSignal(AXIS, np.cos)
        assert result.expression is None
        assert_close(result.data, AXIS.values**2 + np.cos(AXIS.values))


class TestPolynomialSignalArithmetic:
    """Test that monomials stay monomials under scaling."""

    @given(scalar=scalars)
    def test_scaling(self, scalar):
        """Test multiplication and division by scalars."""
        signal = PolynomialSignal(AXIS, 0.5, 3)

        for result, alpha in [
            (scalar * signal, 0.5 * scalar),
            (signal * scalar, 0.5 * scalar),
            (signal / scalar, 0.5 / scalar),
            (-signal, -0.5),
        ]:
            assert type(result) is PolynomialSignal
            assert result.power == 3
            assert_close(result.alpha, alpha)
            assert_close(result.data, alpha * AXIS.values**3)

    def test_quadratic_signals_stay_quadratic(self):
        """Test that scaled quadratic signals keep their class and effective alpha."""
        signal = QuadraticSignal(AXIS, 0.3)
        scaled = -2 * signal

        assert type(scaled) is QuadraticSignal
        assert_close(scaled.effective_alpha, -2 * signal.effective_alpha)

    def test_other_operations_are_algebraic(self):
        """Test that other operations give general algebraic signals."""
        signal = QuadraticSignal(AXIS, 0.3)

        for result in [signal + 1, 1 / (signal + 1), signal * signal]:
            assert type(result) is AlgebraicSignal
        assert_close((signal + 1).data, 0.3 * AXIS.values**2 + 1)
