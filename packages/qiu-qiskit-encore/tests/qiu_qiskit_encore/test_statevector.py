"""Unit tests for statevector.py."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector


class TestValidatedStatevector:
    """Test the validation of the states."""

    def test_accepts_statevectors_and_amplitudes(self):
        """Test that statevectors and array-likes are accepted and copied."""
        amplitudes = np.array([0.6, 0.8])
        statevector = validated_statevector(amplitudes)
        assert isinstance(statevector, Statevector)
        assert statevector.data is not amplitudes
        assert validated_statevector(statevector) == statevector

    @pytest.mark.parametrize("state", [[1.0, 1.0], [0.0, 0.0], [0.5, 0.5, 0.5, 0.6]])
    def test_rejects_unnormalized_states(self, state):
        """Test that unnormalized states are rejected."""
        with pytest.raises(ValueError, match="normalized"):
            validated_statevector(state)

    @pytest.mark.parametrize("state", [[1.0], [1.0, 0.0, 0.0]])
    def test_rejects_non_qubit_states(self, state):
        """Test that the dimension must be a power of 2 of at least one qubit."""
        with pytest.raises(ValueError, match="qubit"):
            validated_statevector(state)
