"""Unit tests for hypothesis_strategies.py."""

import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from python_pytest_helper.assertions import assert_close, is_close
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.constants import MAX_QUBITS, MIN_QUBITS
from qiskit_pytest_helper.hypothesis_strategies import (
    non_normalized_quantum_state_array,
    normalized_quantum_state_array,
    quantum_state_array,
    qubit_axes,
    qubit_sizes,
    valid_qiskit_statevector,
)
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import AxisDomain, PhysicalAxis

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
    assert_close(norm, 1.0)  # Ensure norm is approximately 1


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
    assert not is_close(norm, 1.0)  # Ensure norm is not close to 1


@given(statevector=valid_qiskit_statevector())
def test_valid_qiskit_statevector(statevector: Statevector):
    """Test the valid_qiskit_statevector strategy."""
    assert isinstance(statevector, Statevector)  # Ensure it's a Qiskit Statevector
    assert statevector.is_valid()  # Ensure statevector is normalized and valid
    assert statevector.num_qubits is not None  # Ensure num_qubits is defined
    assert statevector.num_qubits >= 1  # Ensure num_qubits is at least 1


@pytest.mark.parametrize("domain", list(AxisDomain))
@given(data=st.data())
def test_qubit_axes(domain: AxisDomain, data: st.DataObject):
    """Test that the axes have 2**n samples for n within the qubits, in the domain."""
    axis = data.draw(qubit_axes(domain, orderings=st.just(IndexOrdering.FFT)))
    assert isinstance(axis, PhysicalAxis)
    assert axis.size in {2**n for n in range(MIN_QUBITS, MAX_QUBITS + 1)}
    assert axis.domain is domain
    assert axis.ordering is IndexOrdering.FFT


@given(size=qubit_sizes(min_qubits=1, max_qubits=2))
def test_qubit_sizes(size: int):
    """Test that the sizes are the powers of 2 of the numbers of qubits."""
    assert size in {2, 4}
