"""Arithmetic operators shared by the signal classes."""

import operator
from collections.abc import Callable
from numbers import Number
from typing import Any

BinaryOperator = Callable[[Any, Any], Any]
"""An elementwise binary operator, e.g. `operator.add`."""


def is_scalar(value: object) -> bool:
    """Whether the value is a scalar number, e.g. a Python or NumPy number."""
    return isinstance(value, Number)


def require_same_axis(axis: object, other_axis: object) -> None:
    """Raise if two signals to be combined are not sampled on equal axes."""
    if axis != other_axis:
        raise ValueError(
            f"Signals can only be combined on equal axes, got {axis!r} and "
            f"{other_axis!r}."
        )


class ArithmeticOperators:
    """Mixin implementing the arithmetic operators through a single `_binary`.

    Subclasses implement `_binary(other, op, reflected)`, which combines the signal
    with `other` by the elementwise operator `op`, as `op(self, other)` or, if
    `reflected`, as `op(other, self)`. It returns `NotImplemented` for unsupported
    operands, so that Python tries the other operand's operator.
    """

    __array_ufunc__ = None
    """Make NumPy arrays and scalars defer to the operators of the signal."""

    def _binary(self, other: Any, op: BinaryOperator, reflected: bool) -> Any:
        raise NotImplementedError

    def __add__(self, other: Any) -> Any:
        """Return `self + other`."""
        return self._binary(other, operator.add, reflected=False)

    def __radd__(self, other: Any) -> Any:
        """Return `other + self`."""
        return self._binary(other, operator.add, reflected=True)

    def __sub__(self, other: Any) -> Any:
        """Return `self - other`."""
        return self._binary(other, operator.sub, reflected=False)

    def __rsub__(self, other: Any) -> Any:
        """Return `other - self`."""
        return self._binary(other, operator.sub, reflected=True)

    def __mul__(self, other: Any) -> Any:
        """Return `self * other`."""
        return self._binary(other, operator.mul, reflected=False)

    def __rmul__(self, other: Any) -> Any:
        """Return `other * self`."""
        return self._binary(other, operator.mul, reflected=True)

    def __truediv__(self, other: Any) -> Any:
        """Return `self / other`."""
        return self._binary(other, operator.truediv, reflected=False)

    def __rtruediv__(self, other: Any) -> Any:
        """Return `other / self`."""
        return self._binary(other, operator.truediv, reflected=True)

    def __pow__(self, other: Any) -> Any:
        """Return `self ** other`."""
        return self._binary(other, operator.pow, reflected=False)

    def __rpow__(self, other: Any) -> Any:
        """Return `other ** self`."""
        return self._binary(other, operator.pow, reflected=True)

    def __neg__(self) -> Any:
        """Return `-self`."""
        return self._binary(-1, operator.mul, reflected=True)

    def __pos__(self) -> Any:
        """Return `+self`, the signal itself."""
        return self
