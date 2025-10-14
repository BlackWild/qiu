"""Strategies for generating quantum states."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from qiskit.quantum_info import Statevector
from qiskit_signals.helper_types import AxisType, EncodingType
from qiskit_signals.quantum_axis import MomentumAxis, PositionAxis
from qiskit_signals.quantum_signal import GenericQuantumSignal, PolynomialQuantumSignal

from qiskit_pytest_helper.constants import (
    MAX_MAGNITUDE,
    MAX_QUBITS,
    MIN_MAGNITUDE,
    MIN_QUBITS,
)


@st.composite
def quantum_state_array(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
) -> npt.NDArray[np.complex128]:
    """A strategy for generating quantum states as a NumPy array."""
    num_qubits = draw(st.integers(min_qubits, max_qubits))
    dim = 2**num_qubits
    state = draw(
        arrays(
            dtype=np.complex128,
            shape=dim,
            elements=st.complex_numbers(
                min_magnitude=min_magnitude,
                max_magnitude=max_magnitude,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return state


@st.composite
def normalized_quantum_state_array(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
) -> npt.NDArray[np.complex128]:
    """A strategy for generating normalized quantum states as a NumPy array."""
    state = draw(
        quantum_state_array(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
        )
    )
    return state / np.linalg.norm(state)


@st.composite
def non_normalized_quantum_state_array(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
) -> npt.NDArray[np.complex128]:
    """A strategy for generating non-normalized quantum states as a NumPy array."""
    # generate a normalized state first, then scale it
    state = draw(
        normalized_quantum_state_array(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
        )
    )
    scale = draw(
        st.complex_numbers(
            min_magnitude=0.01,
            max_magnitude=MAX_MAGNITUDE,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    if np.isclose(np.abs(scale), 1.0):
        scale += 5  # ensure it's not normalized

    scaled_state = scale * state
    return scaled_state


@st.composite
def valid_qiskit_statevector(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
) -> Statevector:
    """A strategy for generating valid quantum states as a NumPy array."""
    state = draw(
        normalized_quantum_state_array(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
        )
    )
    statevector = Statevector(state)
    return statevector


@st.composite
def state_pairs_with_equal_qubits(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
) -> tuple[Statevector, Statevector]:
    """A strategy for generating pairs of quantum states with the same number of qubits."""
    num_qubits = draw(st.integers(min_qubits, max_qubits))
    state1 = draw(
        normalized_quantum_state_array(
            min_qubits=num_qubits,
            max_qubits=num_qubits,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
        )
    )
    state2 = draw(
        normalized_quantum_state_array(
            min_qubits=num_qubits,
            max_qubits=num_qubits,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
        )
    )
    return Statevector(state1), Statevector(state2)


@st.composite
def position_axis(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    forced_encoding: EncodingType | None = None,
) -> PositionAxis:
    """A strategy for generating PositionAxis objects."""
    num_qubits = draw(st.integers(min_qubits, max_qubits))
    delta_x = draw(
        st.floats(
            min_value=MIN_MAGNITUDE,
            max_value=1,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    encoding = forced_encoding or draw(st.sampled_from(EncodingType.list()))

    return PositionAxis(num_qubits=num_qubits, delta_x=delta_x, encoding=encoding)


@st.composite
def momentum_axis(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    forced_encoding: EncodingType | None = None,
) -> MomentumAxis:
    """A strategy for generating MomentumAxis objects."""
    num_qubits = draw(st.integers(min_qubits, max_qubits))
    delta_x = draw(
        st.floats(
            min_value=MIN_MAGNITUDE,
            max_value=1,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    encoding = forced_encoding or draw(st.sampled_from(EncodingType.list()))

    return MomentumAxis(
        num_qubits=num_qubits, delta_x=delta_x, encoding=encoding, hbar=1.0
    )


@st.composite
def random_positive_signal(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
    forced_sum_value: float = 1.0,
) -> GenericQuantumSignal:
    """A strategy for generating random signals."""

    axis = draw(position_axis(min_qubits=min_qubits, max_qubits=max_qubits))

    # Generate random coefficients for a polynomial signal
    degree = draw(st.integers(min_value=1, max_value=5))
    coefficients = draw(
        arrays(
            dtype=np.float64,
            shape=degree + 1,
            elements=st.floats(
                min_value=min_magnitude,
                max_value=max_magnitude,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    def signal_function(x: npt.NDArray) -> npt.NDArray:
        """A polynomial signal function."""
        return np.abs(sum(c * x**i for i, c in enumerate(coefficients)))

    max = np.max(signal_function(axis.axis_values))

    # Scale the function to have the desired sum value
    def normalized_signal_function(x):
        return (forced_sum_value / max) * signal_function(x)

    signal = GenericQuantumSignal(axis=axis, signal_function=normalized_signal_function)

    return signal


@st.composite
def random_polynomial_signal(
    draw,
    degree: int,
    axis_type: AxisType,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
    forced_encoding: EncodingType | None = None,
) -> PolynomialQuantumSignal:
    """A strategy for generating random polynomial signals."""

    axis = (
        draw(
            position_axis(
                min_qubits=min_qubits,
                max_qubits=max_qubits,
                forced_encoding=forced_encoding,
            )
        )
        if axis_type == AxisType.POSITION
        else draw(
            momentum_axis(
                min_qubits=min_qubits,
                max_qubits=max_qubits,
                forced_encoding=forced_encoding,
            )
        )
    )

    # Generate random coefficient for the polynomial signal
    coefficient = draw(
        st.floats(
            min_value=min_magnitude,
            max_value=max_magnitude,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    signal = PolynomialQuantumSignal(axis=axis, alpha=coefficient, power=degree)

    return signal
