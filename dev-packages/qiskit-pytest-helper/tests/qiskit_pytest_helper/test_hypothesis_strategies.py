"""Unit tests for hypothesis_strategies.py."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis import given
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.hypothesis_strategies import (
    non_normalized_quantum_state_array,
    normalized_quantum_state_array,
    position_axis,
    quantum_state_array,
    random_positive_signal,
    valid_qiskit_statevector,
)
from qiskit_signals.helper_types import AxisType
from qiskit_signals.quantum_axis import PositionAxis
from qiskit_signals.quantum_signal import GenericQuantumSignal
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol

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


@given(axis=position_axis())
def test_position_axis(axis: PositionAxis):
    """Test the position_axis strategy."""
    assert isinstance(axis, PositionAxis)  # Ensure it's a PositionAxis instance


@given(signal=random_positive_signal(axis_type=AxisType.POSITION))
def test_random_signal(signal: GenericQuantumSignal):
    """Test the random_signal strategy."""
    assert isinstance(
        signal, GenericQuantumSignal
    )  # Ensure it's a GenericQuantumSignal
    data = signal.data
    assert isinstance(data, np.ndarray)  # Ensure data is a NumPy array
    assert data.ndim == 1  # Ensure data is one-dimensional
    assert data.size >= 2  # Ensure data has at least two elements
    assert np.isfinite(data).all()  # Ensure all elements are finite
    assert not np.isnan(data).any()  # Ensure no NaNs

    axis = signal.axis
    assert isinstance(axis, PositionAxis)  # Ensure axis is a PositionAxis instance
    assert axis.dimension == data.size  # Ensure axis dimension matches data size

    assert (data >= 0).all()  # Ensure all elements are non-negative
    assert np.isclose(np.sum(data), 1.0)  # Ensure sum of elements is close to 1.0
