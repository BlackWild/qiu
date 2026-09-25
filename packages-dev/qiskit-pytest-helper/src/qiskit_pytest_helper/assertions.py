"""Assertions comparing quantum states and operators with Qiskit's equality.

The comparisons are exact, global phase included, up to the default tolerances of
Qiskit's `Statevector` and `Operator` equality, instead of tolerances chosen per test.
Circuits are compared by what they do: the state they prepare from `|0...0>`, or
their unitary.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import Operator, Statevector, state_fidelity


def assert_equal_states(actual: Any, expected: Any) -> None:
    """Assert that two states are equal, including their global phase.

    Args:
        actual: The state, as a `Statevector`, its amplitudes or a circuit preparing
            it from `|0...0>`.
        expected: The expected state, in any of the same forms.
    """
    actual_state, expected_state = Statevector(actual), Statevector(expected)
    assert actual_state.dims() == expected_state.dims(), (
        f"The states have the dimensions {actual_state.dims()} and "
        f"{expected_state.dims()}."
    )
    assert actual_state == expected_state, (
        "The states differ: the largest amplitude deviation is "
        f"{_max_deviation(actual_state.data, expected_state.data)}, the fidelity is "
        f"{state_fidelity(actual_state, expected_state, validate=False)}.\n"
        f"actual:   {actual_state.data}\nexpected: {expected_state.data}"
    )


def assert_equal_operators(actual: Any, expected: Any) -> None:
    """Assert that two operators are equal, including their global phase.

    Args:
        actual: The operator, as an `Operator`, its matrix, a gate or a circuit.
        expected: The expected operator, in any of the same forms.
    """
    actual_operator, expected_operator = Operator(actual), Operator(expected)
    assert actual_operator.dim == expected_operator.dim, (
        f"The operators have the dimensions {actual_operator.dim} and "
        f"{expected_operator.dim}."
    )
    assert actual_operator == expected_operator, (
        "The operators differ: the largest entry deviation is "
        f"{_max_deviation(actual_operator.data, expected_operator.data)}; up to a "
        "global phase, they are "
        f"{'equal' if actual_operator.equiv(expected_operator) else 'different'}."
    )


def assert_unitary(operator: Any) -> None:
    """Assert that an operator, e.g. a matrix, is unitary."""
    assert Operator(operator).is_unitary(), f"Not unitary:\n{Operator(operator).data}"


def _max_deviation(actual: npt.ArrayLike, expected: npt.ArrayLike) -> float:
    """Return the largest absolute deviation of two arrays of the same shape."""
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
