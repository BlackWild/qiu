"""Strategies for generating quantum states."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from qiskit.quantum_info import Statevector

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
