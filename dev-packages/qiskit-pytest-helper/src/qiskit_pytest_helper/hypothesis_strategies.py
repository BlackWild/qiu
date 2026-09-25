"""Hypothesis strategies for quantum states and for axes representable by qubits.

The strategies of numbers, axes and signals themselves are the ones of
`python_pytest_helper.hypothesis_strategies`, e.g. signals on qubit axes are
`positive_polynomial_signals(qubit_axes(AxisDomain.POSITION))`.
"""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from python_pytest_helper.assertions import is_close
from python_pytest_helper.hypothesis_strategies import (
    axis_spacings,
    index_orderings,
    physical_axes,
    power_of_two_sizes,
)
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain, PhysicalAxis
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
    if is_close(np.abs(scale), 1.0):
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
    """A strategy for normalized quantum states as Qiskit `Statevector`s."""
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


moderate_alphas = st.floats(min_value=MIN_MAGNITUDE, max_value=MAX_MAGNITUDE)
"""A strategy for coefficients of monomial signals keeping their phases moderate."""


def qubit_sizes(
    min_qubits: int = MIN_QUBITS, max_qubits: int = MAX_QUBITS
) -> st.SearchStrategy[int]:
    """A strategy for the numbers of samples `2**n` representable by `n` qubits."""
    return power_of_two_sizes(min_exponent=min_qubits, max_exponent=max_qubits)


def qubit_axes(
    domain: AxisDomain = AxisDomain.POSITION,
    min_qubits: int = MIN_QUBITS,
    max_qubits: int = MAX_QUBITS,
    orderings: st.SearchStrategy[IndexOrdering] = index_orderings,
) -> st.SearchStrategy[PhysicalAxis]:
    """A strategy for axes of `2**n` samples, representable by `n` qubits.

    Fourier axes are conjugate to a position axis, momenta with `hbar = 1`, see
    `python_pytest_helper.hypothesis_strategies.physical_axes`. The position spacings
    are at most 1, keeping the phases of polynomial signals moderate.

    Returns:
        The strategy of the axes.
    """
    return physical_axes(
        domain,
        sizes=qubit_sizes(min_qubits, max_qubits),
        spacings=axis_spacings(min_value=MIN_MAGNITUDE, max_value=1.0),
        orderings=orderings,
    )
