"""Unit tests for hypothesis_strategies.py."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
from hypothesis import given
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.hypothesis_strategies import (
    non_normalized_quantum_state_array,
    normalized_quantum_state_array,
    quantum_state_array,
    valid_qiskit_statevector,
)


@given(quantum_state=quantum_state_array(min_qubits=1, max_qubits=5))
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


@given(quantum_state=normalized_quantum_state_array(min_qubits=1, max_qubits=5))
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


@given(quantum_state=non_normalized_quantum_state_array(min_qubits=1, max_qubits=5))
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


@given(statevector=valid_qiskit_statevector(min_qubits=1, max_qubits=5))
def test_valid_qiskit_statevector(statevector: Statevector):
    """Test the valid_qiskit_statevector strategy."""
    assert isinstance(statevector, Statevector)  # Ensure it's a Qiskit Statevector
    assert statevector.is_valid()  # Ensure statevector is normalized and valid
    assert statevector.num_qubits is not None  # Ensure num_qubits is defined
    assert statevector.num_qubits >= 1  # Ensure num_qubits is at least 1
