"""Unit tests for algebraic_signal.py."""

import builtins
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_signals.algebraic_signal import (
    AlgebraicSignal,
    PolynomialSignal,
    QuadraticSignal,
)
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PhysicalAxis, PositionAxis
from python_signals.signal import Signal


@pytest.fixture(scope="module")
def sympy():
    """The SymPy module, skipping the test if SymPy is not installed."""
    return pytest.importorskip("sympy")


@st.composite
def position_axes(draw) -> PositionAxis:
    """A strategy for generating position axes."""
    return PositionAxis(
        size=draw(st.integers(min_value=1, max_value=64)),
        delta_x=draw(st.floats(min_value=1e-3, max_value=10.0)),
        ordering=draw(st.sampled_from(list(IndexOrdering))),
    )


alphas = st.floats(min_value=-10.0, max_value=10.0)

UNDERFLOW_ATOL = float(np.finfo(np.float64).tiny)
"""The smallest normal float, below which floats lose their relative precision.

Monomials of tiny coefficients underflow to subnormal values, which only agree
absolutely at this scale.
"""
powers = st.integers(min_value=0, max_value=4)


class TestAlgebraicSignalFromFunction:
    """Test AlgebraicSignal created from a Python callable."""

    @given(axis=position_axes())
    def test_essentials(self, axis: PhysicalAxis):
        """Test the essential properties of an algebraic signal."""

        def function(x):
            return np.cos(x) + x**2

        signal = AlgebraicSignal(axis, function)

        assert signal.axis is axis
        assert signal.function is function
        assert signal.expression is None
        assert signal.symbol is None
        assert signal.size == axis.size
        np.testing.assert_array_equal(signal.data, function(axis.values))

    def test_data_is_computed_once(self):
        """Test that the function is only evaluated once on the axis."""
        calls = []

        def function(x):
            calls.append(x)
            return x

        signal = AlgebraicSignal(PositionAxis(8, 1.0, IndexOrdering.NATURAL), function)
        assert signal.data is signal.data
        assert len(calls) == 1

    def test_call_evaluates_at_arbitrary_values(self):
        """Test that the signal can be evaluated off its axis."""
        signal = AlgebraicSignal(PositionAxis(4, 1.0, IndexOrdering.NATURAL), np.sin)
        values = np.linspace(-1, 1, 7)
        np.testing.assert_array_equal(signal(values), np.sin(values))

    def test_constant_function_is_broadcast(self):
        """Test that a function returning a scalar gives one value per sample."""
        axis = PositionAxis(5, 1.0, IndexOrdering.CENTERED)
        signal = AlgebraicSignal(axis, lambda x: 2.0)
        np.testing.assert_array_equal(signal.data, np.full(5, 2.0))

    def test_complex_function(self):
        """Test that complex-valued functions are supported."""
        axis = PositionAxis(8, 0.5, IndexOrdering.FFT)
        signal = AlgebraicSignal(axis, lambda x: np.exp(1j * x))
        assert np.iscomplexobj(signal.data)
        np.testing.assert_allclose(np.abs(signal.data), 1.0)

    @given(axis=position_axes())
    def test_to_signal(self, axis: PhysicalAxis):
        """Test the conversion to a sampled signal."""
        algebraic = AlgebraicSignal(axis, lambda x: 1 + x**2)
        signal = algebraic.to_signal()

        assert type(signal) is Signal
        assert signal.axis is axis
        np.testing.assert_array_equal(signal.data, algebraic.data)
        assert np.isclose(np.linalg.norm(signal.normalized_data), 1.0)


class TestAlgebraicSignalFromSympy:
    """Test AlgebraicSignal created from a SymPy expression."""

    @given(axis=position_axes(), alpha=alphas)
    def test_matches_the_numpy_function(self, sympy, axis: PhysicalAxis, alpha: float):
        """Test that the expression evaluates like its NumPy counterpart."""
        x = sympy.Symbol("x")
        signal = AlgebraicSignal.from_sympy(
            axis, alpha * sympy.exp(-(x**2)) + sympy.cos(x)
        )

        assert signal.symbol == x
        np.testing.assert_allclose(
            signal.data, alpha * np.exp(-(axis.values**2)) + np.cos(axis.values)
        )

    def test_keeps_the_expression(self, sympy):
        """Test that the symbolic expression can be inspected and manipulated."""
        x = sympy.Symbol("x")
        signal = AlgebraicSignal.from_sympy(
            PositionAxis(4, 1.0, IndexOrdering.NATURAL), 3 * x**2 - x
        )

        assert signal.expression == 3 * x**2 - x
        assert sympy.degree(signal.expression, x) == 2
        assert sympy.diff(signal.expression, x) == 6 * x - 1
        assert "3*x**2 - x" in repr(signal)

    def test_from_string(self, sympy):
        """Test that the expression can be given as a string."""
        axis = PositionAxis(6, 0.5, IndexOrdering.CENTERED)
        signal = AlgebraicSignal.from_sympy(axis, "2*t**3")
        np.testing.assert_allclose(signal.data, 2 * axis.values**3)
        assert signal.symbol == sympy.Symbol("t")

    def test_explicit_symbol(self, sympy):
        """Test that the symbol of the axis can be chosen explicitly."""
        x, alpha = sympy.symbols("x alpha")
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        signal = AlgebraicSignal.from_sympy(axis, (alpha * x**2).subs(alpha, 3), x)
        np.testing.assert_allclose(signal.data, 3 * axis.values**2)

    def test_constant_expression(self, sympy):
        """Test that constant expressions give one value per sample."""
        axis = PositionAxis(5, 1.0, IndexOrdering.NATURAL)
        signal = AlgebraicSignal.from_sympy(axis, sympy.Integer(7))
        np.testing.assert_array_equal(signal.data, np.full(5, 7))

    def test_complex_expression(self, sympy):
        """Test that complex expressions evaluate to complex values."""
        x = sympy.Symbol("x")
        axis = PositionAxis(8, 0.25, IndexOrdering.FFT)
        signal = AlgebraicSignal.from_sympy(axis, sympy.exp(sympy.I * x))
        np.testing.assert_allclose(signal.data, np.exp(1j * axis.values))

    @pytest.mark.parametrize("expression", ["x > 1", "Eq(x, 1)"])
    def test_non_algebraic_expression(self, sympy, expression: str):
        """Test that relations and other non-algebraic objects are rejected."""
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(TypeError, match="algebraic expression"):
            AlgebraicSignal.from_sympy(axis, expression)

    def test_ambiguous_symbols(self, sympy):
        """Test that the symbol must be given if there are several free symbols."""
        x, y = sympy.symbols("x y")
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(ValueError, match="free symbols"):
            AlgebraicSignal.from_sympy(axis, x * y)

    def test_unbound_symbols(self, sympy):
        """Test that the expression must not have other free symbols."""
        x, y = sympy.symbols("x y")
        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(ValueError, match="other than x"):
            AlgebraicSignal.from_sympy(axis, x * y, x)


class TestAlgebraicSignalWithoutSympy:
    """Test AlgebraicSignal.from_sympy without SymPy installed."""

    def test_missing_sympy(self, monkeypatch: pytest.MonkeyPatch):
        """Test the error message if SymPy is not installed."""
        real_import = builtins.__import__

        def import_without_sympy(name, *args, **kwargs):
            if name == "sympy":
                raise ImportError("No module named 'sympy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "sympy", raising=False)
        monkeypatch.setattr(builtins, "__import__", import_without_sympy)

        axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
        with pytest.raises(ImportError, match=r"python-signals\[sympy\]"):
            AlgebraicSignal.from_sympy(axis, "x")


class TestPolynomialSignal:
    """Test the PolynomialSignal."""

    @given(axis=position_axes(), alpha=alphas, power=powers)
    def test_essentials(self, axis: PhysicalAxis, alpha: float, power: int):
        """Test the essential properties of a polynomial signal."""
        signal = PolynomialSignal(axis, alpha, power)

        assert isinstance(signal, AlgebraicSignal)
        assert signal.axis is axis
        assert signal.alpha == alpha
        assert signal.power == power
        np.testing.assert_allclose(signal.data, alpha * axis.values**power)

    @given(axis=position_axes(), alpha=alphas, power=powers)
    def test_effective_alpha(self, axis: PhysicalAxis, alpha: float, power: int):
        """Test that the signal is effective_alpha times the powers of the indices."""
        signal = PolynomialSignal(axis, alpha, power)
        np.testing.assert_allclose(
            signal.data,
            signal.effective_alpha * axis.index.astype(float) ** power,
            atol=UNDERFLOW_ATOL,
        )

    @given(axis=position_axes(), alpha=alphas, power=powers)
    def test_matches_sympy(self, sympy, axis: PhysicalAxis, alpha: float, power: int):
        """Test that the monomial matches its symbolic counterpart."""
        x = sympy.Symbol("x")
        symbolic = AlgebraicSignal.from_sympy(axis, alpha * x**power, x)
        np.testing.assert_allclose(
            PolynomialSignal(axis, alpha, power).data,
            symbolic.data,
            atol=UNDERFLOW_ATOL,
        )


class TestQuadraticSignal:
    """Test the QuadraticSignal."""

    @given(axis=position_axes(), alpha=alphas)
    def test_essentials(self, axis: PhysicalAxis, alpha: float):
        """Test that a quadratic signal is the polynomial signal of power 2."""
        signal = QuadraticSignal(axis, alpha)

        assert isinstance(signal, PolynomialSignal)
        assert signal.power == 2
        assert signal.alpha == alpha
        np.testing.assert_allclose(signal.data, PolynomialSignal(axis, alpha, 2).data)
