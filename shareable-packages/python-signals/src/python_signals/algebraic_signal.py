"""Signals given by an algebraic expression evaluated on a physical axis."""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from python_signals.arithmetic import (
    ArithmeticOperators,
    BinaryOperator,
    is_scalar,
    require_same_axis,
)
from python_signals.signal import Signal

if TYPE_CHECKING:
    import sympy

    from python_signals.physical_axis import PhysicalAxis

SignalFunctionType = Callable[[npt.NDArray[Any]], npt.ArrayLike]
"""A vectorized function, mapping an array of axis values to the signal values.

It is called once with all values as an array, e.g. `axis.values`, so it must act
elementwise on arrays, e.g. with NumPy functions. The dtype of the array is not fixed,
so that functions annotated for a specific dtype are accepted. It may return a scalar
for constant signals, which is broadcast to the axis values.
"""


class AlgebraicSignal(ArithmeticOperators):
    """A signal given by an algebraic expression of the axis values.

    The expression is held as a vectorized function, mapping an array of axis values
    to the array of signal values. It is either given directly, e.g. as a lambda or
    a NumPy function, or compiled from a SymPy expression with `from_sympy`, which
    also keeps the symbolic expression.

    Algebraic signals support the arithmetic operators `+`, `-`, `*`, `/` and `**`
    with scalars and with algebraic signals on an equal axis, composing their
    functions, and their SymPy expressions if both operands have one. Combined with a
    sampled `Signal`, they are sampled first, and the result is a sampled `Signal`.
    """

    axis: PhysicalAxis
    """The axis the signal is evaluated on."""
    function: SignalFunctionType
    """The vectorized function mapping axis values to signal values."""
    expression: sympy.Expr | None
    """The SymPy expression of the signal, if it was created from one."""
    symbol: sympy.Symbol | None
    """The SymPy symbol standing for the axis values, if created from an expression."""

    def __init__(self, axis: PhysicalAxis, function: SignalFunctionType) -> None:
        """Initialize the signal from a vectorized function.

        Args:
            axis: The axis the signal is evaluated on.
            function: A function mapping an array of axis values to the array of
                signal values, e.g. `lambda x: np.exp(-(x**2))`.
        """
        self.axis = axis
        self.function = function
        self.expression = None
        self.symbol = None

    @classmethod
    def from_sympy(
        cls,
        axis: PhysicalAxis,
        expression: sympy.Expr | str,
        symbol: sympy.Symbol | None = None,
    ) -> AlgebraicSignal:
        """Create a signal from a SymPy expression in one symbol.

        Requires SymPy, e.g. via the `sympy` extra of this package.

        Args:
            axis: The axis the signal is evaluated on.
            expression: The expression of the signal, or a string SymPy can parse.
            symbol: The symbol standing for the axis values. Can be omitted if the
                expression has at most one free symbol.

        Returns:
            The signal, evaluating the expression with NumPy.

        Raises:
            ImportError: If SymPy is not installed.
            TypeError: If the expression is not an algebraic SymPy expression.
            ValueError: If the expression has free symbols other than `symbol`, or,
                with `symbol` omitted, more than one free symbol or a free symbol
                that is not a plain SymPy symbol.
        """
        try:
            import sympy
        except ImportError as error:
            raise ImportError(
                "AlgebraicSignal.from_sympy requires SymPy, e.g. install "
                "python-signals[sympy]."
            ) from error

        parsed = sympy.sympify(expression)
        if not isinstance(parsed, sympy.Expr):
            raise TypeError(
                f"Expected an algebraic expression, got {parsed} of type "
                f"{type(parsed).__name__}."
            )
        free_symbols = parsed.free_symbols

        if symbol is not None:
            if free_symbols - {symbol}:
                raise ValueError(
                    f"The expression {parsed} has free symbols other than {symbol}: "
                    f"{sorted(map(str, free_symbols - {symbol}))}."
                )
            axis_symbol = symbol
        elif not free_symbols:
            axis_symbol = sympy.Symbol("x")
        elif len(free_symbols) > 1:
            raise ValueError(
                f"The expression {parsed} has the free symbols "
                f"{sorted(map(str, free_symbols))}; give the one standing for "
                "the axis values as `symbol`."
            )
        else:
            (free_symbol,) = free_symbols
            if not isinstance(free_symbol, sympy.Symbol):
                raise ValueError(
                    f"The free symbol {free_symbol} of the expression {parsed} is "
                    "not a plain symbol; give the symbol of the axis values."
                )
            axis_symbol = free_symbol

        signal = cls(axis, sympy.lambdify(axis_symbol, parsed, modules="numpy"))
        signal.expression = parsed
        signal.symbol = axis_symbol
        return signal

    def __repr__(self) -> str:
        """Return a readable representation of the signal."""
        definition = self.expression if self.expression is not None else self.function
        return f"{type(self).__name__}(axis={self.axis!r}, expression={definition})"

    def __call__(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Evaluate the signal at arbitrary axis values.

        Constant expressions are broadcast to the shape of the values.

        Returns:
            The signal values, an array of the shape of the axis values.
        """
        values = np.asarray(values)
        result = np.asarray(self.function(values))
        return np.broadcast_to(result, values.shape).copy()

    @cached_property
    def data(self) -> npt.NDArray[np.number]:
        """Return the signal evaluated on the axis values."""
        return self(self.axis.values)

    @property
    def size(self) -> int:
        """Return the number of samples of the signal on its axis."""
        return self.axis.size

    def to_signal(self) -> Signal:
        """Return the signal sampled on its axis."""
        return Signal(axis=self.axis, data=self.data)

    def _binary(self, other: Any, op: BinaryOperator, reflected: bool) -> Any:
        """Combine with a scalar, an algebraic signal or a sampled signal."""
        if isinstance(other, Signal):
            sampled = self.to_signal()
            return op(other, sampled) if reflected else op(sampled, other)

        if is_scalar(other):
            other_function = _constant_function(other)
            other_expression = other
        elif isinstance(other, AlgebraicSignal):
            require_same_axis(self.axis, other.axis)
            other_function = other.function
            other_expression = other._expression_in(self.symbol)
        else:
            return NotImplemented

        left, right = (
            (other_function, self.function)
            if reflected
            else (self.function, other_function)
        )
        result = AlgebraicSignal(
            self.axis, lambda x: op(np.asarray(left(x)), np.asarray(right(x)))
        )
        if self.expression is not None and other_expression is not None:
            result.expression = (
                op(other_expression, self.expression)
                if reflected
                else op(self.expression, other_expression)
            )
            result.symbol = self.symbol
        return result

    def _expression_in(self, symbol: sympy.Symbol | None) -> sympy.Expr | None:
        """Return the expression in terms of the given symbol, if both exist."""
        if self.expression is None or symbol is None or self.symbol is None:
            return None
        # replacing a symbol by a symbol keeps the expression an expression, which
        # SymPy's annotations do not tell
        return cast("sympy.Expr", self.expression.xreplace({self.symbol: symbol}))


def _constant_function(value: Any) -> SignalFunctionType:
    """Return the function of a constant signal."""
    return lambda x: value


SampledSignal = Signal | AlgebraicSignal
"""A signal with sampled values on its axis, given directly or by an expression.

Both kinds provide the `axis` and the sampled values `data`.
"""


class PolynomialSignal(AlgebraicSignal):
    """A monomial signal of the form f(x) = alpha * x^power."""

    alpha: float
    """The coefficient of the monomial."""
    power: int
    """The power of the monomial."""

    def __init__(self, axis: PhysicalAxis, alpha: float, power: int) -> None:
        """Initialize the polynomial signal."""
        super().__init__(axis=axis, function=lambda x: alpha * x**power)
        self.alpha = alpha
        self.power = power

    def _with_alpha(self, alpha: Any) -> PolynomialSignal:
        """Return the monomial of the same power with another coefficient."""
        return PolynomialSignal(self.axis, alpha, self.power)

    def _binary(self, other: Any, op: BinaryOperator, reflected: bool) -> Any:
        """Keep monomials monomial under scaling, i.e. `*` and `/` by scalars."""
        if is_scalar(other):
            if op is operator.mul:
                return self._with_alpha(self.alpha * other)
            if op is operator.truediv and not reflected:
                return self._with_alpha(self.alpha / other)
        return super()._binary(other, op, reflected)

    @property
    def effective_alpha(self) -> float:
        """Return the coefficient of the monomial in terms of the integer indices.

        Such that the sampled signal is `effective_alpha * axis.index**power`.
        """
        return self.alpha * self.axis.period**self.power


class QuadraticSignal(PolynomialSignal):
    """A quadratic signal of the form f(x) = alpha * x^2. Also called an intensity signal."""

    def __init__(self, axis: PhysicalAxis, alpha: float) -> None:
        """Initialize the quadratic signal."""
        super().__init__(axis=axis, alpha=alpha, power=2)

    def _with_alpha(self, alpha: Any) -> QuadraticSignal:
        """Return the quadratic signal with another coefficient."""
        return QuadraticSignal(self.axis, alpha)
