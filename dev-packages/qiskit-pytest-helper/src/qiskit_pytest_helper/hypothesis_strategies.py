"""Strategies for generating quantum states."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from qiskit.quantum_info import Statevector


@st.composite
def quantum_state_array(
    draw,
    min_qubits=1,
    max_qubits=5,
    min_magnitude=0.001,
    max_magnitude=1000.0,
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
    min_qubits=1,
    max_qubits=5,
    min_magnitude=0.001,
    max_magnitude=1000.0,
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
    min_qubits=1,
    max_qubits=5,
    min_magnitude=0.001,
    max_magnitude=1000.0,
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
            max_magnitude=1000.0,
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
    min_qubits=1,
    max_qubits=5,
    min_magnitude=0.001,
    max_magnitude=1000.0,
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
