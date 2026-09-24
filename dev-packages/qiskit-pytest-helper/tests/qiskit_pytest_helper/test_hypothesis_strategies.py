"""Unit tests for hypothesis_strategies.py."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis import given
from python_signals.algebraic_signal import AlgebraicSignal, PolynomialSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain, MomentumAxis, PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.constants import MAX_QUBITS, MIN_QUBITS
from qiskit_pytest_helper.hypothesis_strategies import (
    momentum_axis,
    non_normalized_quantum_state_array,
    normalized_quantum_state_array,
    position_axis,
    quantum_state_array,
    random_polynomial_signal,
    random_positive_signal,
    valid_qiskit_statevector,
)

# TODO: must rework these tests by manually generating values for the strategies and checking if the generated values are valid and fall in the expected ranges.


@given(quantum_state=quantum_state_array())
def test_quantum_state_essentials(quantum_state: npt.NDArray[np.complex128]):
    """Test the quantum_state_array strategy."""
    assert isinstance(quantum_state, np.ndarray)  # Ensure it's a NumPy array
    assert np.log2(quantum_state.size).is_integer()  # Ensure size is a power of 2
    assert (quantum_state != 0).any()  # Ensure not all elements are zero
    assert np.isfinite(quantum_state).all()  # Ensure all elements are finite
    assert not np.isnan(quantum_state).any()  # Ensure no NaNs

    norm = np.linalg.norm(quantum_state)
    assert np.isfinite(norm)  # Ensure norm is finite
    assert norm > 0  # Ensure norm is positive


@given(quantum_state=normalized_quantum_state_array())
def test_normalized_quantum_state(quantum_state: npt.NDArray[np.complex128]):
    """Test the normalized_quantum_state_array strategy."""
    assert isinstance(quantum_state, np.ndarray)  # Ensure it's a NumPy array
    assert np.log2(quantum_state.size).is_integer()  # Ensure size is a power of 2
    assert (quantum_state != 0).any()  # Ensure not all elements are zero
    assert np.isfinite(quantum_state).all()  # Ensure all elements are finite
    assert not np.isnan(quantum_state).any()  # Ensure no NaNs

    norm = np.linalg.norm(quantum_state)
    assert np.isfinite(norm)  # Ensure norm is finite
    assert np.isclose(norm, 1.0)  # Ensure norm is approximately 1


@given(quantum_state=non_normalized_quantum_state_array())
def test_non_normalized_quantum_state(quantum_state: npt.NDArray[np.complex128]):
    """Test the non_normalized_quantum_state_array strategy."""
    assert isinstance(quantum_state, np.ndarray)  # Ensure it's a NumPy array
    assert np.log2(quantum_state.size).is_integer()  # Ensure size is a power of 2
    assert (quantum_state != 0).any()  # Ensure not all elements are zero
    assert np.isfinite(quantum_state).all()  # Ensure all elements are finite
    assert not np.isnan(quantum_state).any()  # Ensure no NaNs

    norm = np.linalg.norm(quantum_state)
    assert np.isfinite(norm)  # Ensure norm is finite
    assert not np.isclose(norm, 1.0)  # Ensure norm is not close to 1


@given(statevector=valid_qiskit_statevector())
def test_valid_qiskit_statevector(statevector: Statevector):
    """Test the valid_qiskit_statevector strategy."""
    assert isinstance(statevector, Statevector)  # Ensure it's a Qiskit Statevector
    assert statevector.is_valid()  # Ensure statevector is normalized and valid
    assert statevector.num_qubits is not None  # Ensure num_qubits is defined
    assert statevector.num_qubits >= 1  # Ensure num_qubits is at least 1


def is_qubit_sized(size: int) -> bool:
    """Whether the size is 2**n for n between the minimum and maximum qubits."""
    return size in [2**n for n in range(MIN_QUBITS, MAX_QUBITS + 1)]


@given(axis=position_axis())
def test_position_axis(axis: PositionAxis):
    """Test the position_axis strategy."""
    assert isinstance(axis, PositionAxis)
    assert is_qubit_sized(axis.size)
    assert axis.period > 0


@given(axis=momentum_axis(forced_ordering=IndexOrdering.FFT))
def test_momentum_axis(axis: MomentumAxis):
    """Test the momentum_axis strategy with a forced ordering."""
    assert isinstance(axis, MomentumAxis)
    assert is_qubit_sized(axis.size)
    assert axis.ordering is IndexOrdering.FFT
    assert axis.is_fourier_domain


@given(signal=random_positive_signal(domain=AxisDomain.POSITION))
def test_random_positive_signal(signal: AlgebraicSignal):
    """Test the random_positive_signal strategy."""
    assert isinstance(signal, AlgebraicSignal)
    assert isinstance(signal.axis, PositionAxis)

    data = signal.data
    assert data.shape == (signal.axis.size,)
    assert np.isfinite(data).all()
    assert (data >= 0).all()
    assert np.isclose(np.sum(data), 1.0)


@given(signal=random_polynomial_signal(degree=3, domain=AxisDomain.MOMENTUM))
def test_random_polynomial_signal(signal: PolynomialSignal):
    """Test the random_polynomial_signal strategy."""
    assert isinstance(signal, PolynomialSignal)
    assert signal.power == 3
    assert signal.axis.is_fourier_domain
    np.testing.assert_allclose(signal.data, signal.alpha * signal.axis.values**3)
