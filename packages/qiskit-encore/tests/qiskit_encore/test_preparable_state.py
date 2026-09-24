"""Unit tests for preparable_state.py."""

import dataclasses

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from qiskit.quantum_info import Operator, Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.state_preparation import state_preparation_circuit
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_pytest_helper.assertions import (
    assert_equal_operators,
    assert_equal_states,
)
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector

methods = st.sampled_from([SynthesisMethod.DECOMPOSED, SynthesisMethod.DENSE])


class TestPreparableState:
    """Test the PreparableState."""

    @given(statevector=valid_qiskit_statevector(), method=methods)
    def test_essentials(self, statevector: Statevector, method: SynthesisMethod):
        """Test the essential properties of a preparable state."""
        state = PreparableState(statevector, method)

        assert state.statevector == statevector
        assert state.method is method
        assert state.num_qubits == statevector.num_qubits

    @given(statevector=valid_qiskit_statevector(), method=methods)
    def test_circuits(self, statevector: Statevector, method: SynthesisMethod):
        """Test that the circuits are the state preparation circuits of the method."""
        state = PreparableState(statevector, method)

        assert_equal_operators(
            state.circuit, state_preparation_circuit(statevector, method=method)
        )
        assert_equal_operators(state.inverse_circuit, Operator(state.circuit).adjoint())

    def test_default_method(self):
        """Test that the gate method is the default."""
        assert PreparableState([0.6, 0.8]).method is SynthesisMethod.GATE

    def test_accepts_amplitudes_and_raw_methods(self):
        """Test construction from amplitudes and a raw method value."""
        state = PreparableState([0.6, 0.8j], "dense")  # type: ignore[arg-type]
        assert isinstance(state.statevector, Statevector)
        assert state.method is SynthesisMethod.DENSE

    def test_circuits_are_cached(self):
        """Test that the circuits are only built once."""
        state = PreparableState([0.6, 0.8])
        assert state.circuit is state.circuit
        assert state.inverse_circuit is state.inverse_circuit

    def test_is_immutable(self):
        """Test that the state cannot change under its cached circuits."""
        state = PreparableState([0.6, 0.8])
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.statevector = Statevector([0.8, 0.6])  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.method = SynthesisMethod.DENSE  # type: ignore[misc]

    def test_copies_the_amplitudes(self):
        """Test that later changes of the given amplitudes do not leak in."""
        amplitudes = np.array([0.6, 0.8])
        state = PreparableState(amplitudes)
        amplitudes[:] = [0.8, 0.6]
        assert_equal_states(state.circuit, [0.6, 0.8])

    def test_rejects_invalid_states(self):
        """Test that unnormalized states are rejected."""
        with pytest.raises(ValueError, match="normalized"):
            PreparableState([1.0, 1.0])
