"""Strategies for generating quantum states."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis.extra.numpy import arrays
from python_signals.algebraic_signal import AlgebraicSignal, PolynomialSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import (
    AxisDomain,
    MomentumAxis,
    PhysicalAxis,
    PositionAxis,
)
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


@st.composite
def position_axis(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    forced_ordering: IndexOrdering | None = None,
) -> PositionAxis:
    """A strategy for position axes of `2**n` samples, representable by `n` qubits."""
    num_qubits = draw(st.integers(min_qubits, max_qubits))
    delta_x = draw(st.floats(min_value=MIN_MAGNITUDE, max_value=1))
    ordering = forced_ordering or draw(st.sampled_from(list(IndexOrdering)))

    return PositionAxis(size=2**num_qubits, delta_x=delta_x, ordering=ordering)


@st.composite
def momentum_axis(
    draw,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    forced_ordering: IndexOrdering | None = None,
) -> MomentumAxis:
    """A strategy for momentum axes of `2**n` samples, conjugate to a position axis.

    The momenta are in natural units, `hbar = 1`.
    """
    x_axis = draw(
        position_axis(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            forced_ordering=forced_ordering,
        )
    )
    return MomentumAxis.from_position_axis(x_axis, hbar=1.0, keep_ordering=True)


@st.composite
def qubit_axis(
    draw,
    domain: AxisDomain,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    forced_ordering: IndexOrdering | None = None,
) -> PhysicalAxis:
    """A strategy for position or momentum axes of `2**n` samples."""
    strategy = position_axis if domain == AxisDomain.POSITION else momentum_axis
    return draw(
        strategy(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            forced_ordering=forced_ordering,
        )
    )


@st.composite
def random_positive_signal(
    draw,
    domain: AxisDomain,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
    forced_sum_value: float = 1.0,
    forced_ordering: IndexOrdering | None = None,
) -> AlgebraicSignal:
    """A strategy for positive polynomial signals of degree up to 5.

    The signals are scaled such that their samples sum up to `forced_sum_value`.
    """
    axis = draw(
        qubit_axis(
            domain=domain,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            forced_ordering=forced_ordering,
        )
    )

    degree = draw(st.integers(min_value=1, max_value=5))
    coefficients = draw(
        arrays(
            dtype=np.float64,
            shape=degree + 1,
            elements=st.floats(min_value=min_magnitude, max_value=max_magnitude),
        )
    )

    def polynomial(x: npt.NDArray) -> npt.NDArray:
        return np.abs(sum(c * x**i for i, c in enumerate(coefficients)))

    scale = forced_sum_value / np.sum(polynomial(axis.values))
    return AlgebraicSignal(axis, lambda x: scale * polynomial(x))


@st.composite
def random_polynomial_signal(
    draw,
    degree: int,
    domain: AxisDomain,
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_magnitude=MIN_MAGNITUDE,
    max_magnitude=MAX_MAGNITUDE,
    forced_ordering: IndexOrdering | None = None,
) -> PolynomialSignal:
    """A strategy for monomial signals `alpha * x**degree`."""
    axis = draw(
        qubit_axis(
            domain=domain,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            forced_ordering=forced_ordering,
        )
    )
    alpha = draw(st.floats(min_value=min_magnitude, max_value=max_magnitude))
    return PolynomialSignal(axis=axis, alpha=alpha, power=degree)
