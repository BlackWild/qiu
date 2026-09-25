"""Floating-point comparisons for unit tests, shared across the monorepo.

Values are compared relative to their own magnitude, with the default relative
tolerance of `numpy.testing.assert_allclose`, which covers the rounding errors of the
few operations a test compares.

Below the smallest normal float, `tiny`, floats lose their relative precision: they
underflow to subnormal numbers or to zero. Such values only agree absolutely, at the
scale of `tiny`. If the compared values were scaled up by a factor after they could
underflow, e.g. normalized values multiplied back by the norm, the absolute scale is
`tiny` times that factor, given as `scale`; scaling down still leaves the underflow of
the result itself.

Rounding errors are relative to the magnitude of the computation, so quantities that
are ideally 0, e.g. the imaginary part of a real projection, cannot be compared with
0; compare the whole quantity instead, e.g. the projection with its magnitude.

Vectors computed as a whole, e.g. by FFTs or unitary evolutions, have rounding errors
relative to their norm rather than to each entry, so entries near 0 lose their relative
precision; `assert_close_in_norm` compares such vectors in norm instead.

Quantum states and operators are compared with Qiskit's equality instead, see
`qiskit_pytest_helper.assertions`.
"""

import numpy as np
import numpy.typing as npt

RTOL = 1e-7
"""The relative tolerance, the default of `numpy.testing.assert_allclose`."""


def underflow_atol(
    dtype: type[np.inexact] | np.dtype[np.inexact] = np.float64, scale: float = 1.0
) -> float:
    """Return the absolute tolerance of values that may have underflowed.

    Args:
        dtype: The floating-point or complex type of the values.
        scale: The factor the values were scaled by after they could underflow.

    Returns:
        The smallest normal float of the type, times the scale if it is larger than 1.
    """
    return float(np.finfo(dtype).tiny * max(1.0, abs(scale)))


def assert_close(
    actual: npt.ArrayLike, expected: npt.ArrayLike, *, scale: float = 1.0
) -> None:
    """Assert that numbers or arrays agree elementwise up to rounding.

    Args:
        actual: The computed values.
        expected: The expected values, broadcastable to the computed ones.
        scale: The factor the values were scaled by after they could underflow.
    """
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=RTOL,
        atol=underflow_atol(_comparison_dtype(actual, expected), scale),
    )


def is_close(
    actual: npt.ArrayLike, expected: npt.ArrayLike, *, scale: float = 1.0
) -> bool:
    """Return whether numbers or arrays agree elementwise up to rounding.

    The same comparison as `assert_close`, e.g. to exclude values in strategies.
    """
    return bool(
        np.allclose(
            actual,
            expected,
            rtol=RTOL,
            atol=underflow_atol(_comparison_dtype(actual, expected), scale),
        )
    )


def assert_close_in_norm(actual: npt.ArrayLike, expected: npt.ArrayLike) -> None:
    """Assert that two vectors agree up to rounding relative to their norm.

    That is `||actual - expected|| <= RTOL ||expected||`, in the Euclidean norm.

    Args:
        actual: The computed vector.
        expected: The expected vector, of the same shape.
    """
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    assert actual_array.shape == expected_array.shape, (
        f"The shapes differ: {actual_array.shape} and {expected_array.shape}."
    )
    error = float(np.linalg.norm(actual_array - expected_array))
    norm = float(np.linalg.norm(expected_array))
    assert error <= RTOL * norm + underflow_atol(
        _comparison_dtype(actual_array, expected_array)
    ), (
        f"The vectors differ by {error} in norm, relative {error / norm if norm else error}."
    )


def _comparison_dtype(*values: npt.ArrayLike) -> np.dtype[np.inexact]:
    """Return the inexact type the values are compared in, float64 for exact ones."""
    dtype = np.result_type(*(np.asarray(value) for value in values))
    return dtype if np.issubdtype(dtype, np.inexact) else np.dtype(np.float64)
