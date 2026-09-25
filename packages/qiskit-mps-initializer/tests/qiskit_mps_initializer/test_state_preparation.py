"""Unit tests for state_preparation.py."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal
from qiskit.quantum_info import Operator, Statevector
from qiskit_mps_initializer.state_preparation import mps_state_preparation
from qiskit_phase_propagator.sample_based import sample_based_decomposition
from qiskit_pytest_helper.assertions import assert_equal_operators, assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


def random_state(num_qubits: int, seed: int) -> Statevector:
    """A random normalized state."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    return Statevector(data / np.linalg.norm(data))


class TestExactPreparation:
    """Test the states one layer prepares exactly."""

    @settings(deadline=None)
    @given(state=valid_qiskit_statevector(min_qubits=1, max_qubits=3))
    def test_up_to_three_qubits(self, state: Statevector):
        """Test that one layer prepares states of up to 3 qubits exactly."""
        preparation = mps_state_preparation(state, max_layers=5)

        assert preparation.num_layers == 1
        assert preparation.converged
        assert_equal_states(preparation.circuit, state)

    @pytest.mark.parametrize("num_qubits", [4, 5, 6])
    def test_product_states(self, num_qubits: int):
        """Test that one layer prepares product states, of bond dimension 1."""
        state = Statevector.from_label("+0-1+-"[:num_qubits])
        preparation = mps_state_preparation(state, max_layers=5)

        assert preparation.num_layers == 1
        assert_equal_states(preparation.circuit, state)

    def test_zero_state(self):
        """Test that the all-zero state needs no layer."""
        preparation = mps_state_preparation(Statevector.from_label("0000"), 3)
        assert preparation.num_layers == 0
        assert preparation.converged


class TestApproximatePreparation:
    """Test the preparation of states of higher bond dimension with several layers."""

    @settings(deadline=None, max_examples=20)
    @given(
        num_qubits=st.integers(min_value=4, max_value=5),
        seed=st.integers(0, 2**16),
        tolerance=st.floats(min_value=1e-3, max_value=0.1),
    )
    def test_reaches_the_tolerance(self, num_qubits: int, seed: int, tolerance: float):
        """Test that enough layers prepare the state within the tolerance."""
        state = random_state(num_qubits, seed)
        preparation = mps_state_preparation(state, max_layers=500, tolerance=tolerance)

        assert preparation.converged
        distance = np.linalg.norm(Statevector(preparation.circuit).data - state.data)
        assert distance < tolerance
        assert_close(preparation.error, distance)

    @pytest.mark.parametrize("max_layers", [1, 2, 7])
    def test_max_layers(self, max_layers: int):
        """Test that at most max_layers layers are used, and the error is reported."""
        state = random_state(5, seed=max_layers)
        preparation = mps_state_preparation(state, max_layers=max_layers)

        assert preparation.num_layers == max_layers
        assert not preparation.converged
        distance = np.linalg.norm(Statevector(preparation.circuit).data - state.data)
        assert_close(preparation.error, distance)

    def test_more_layers_are_better(self):
        """Test that the fidelity grows from a few layers to many."""
        state = random_state(5, seed=0)
        fidelities = [
            abs(np.vdot(Statevector(preparation.circuit).data, state.data)) ** 2
            for preparation in (
                mps_state_preparation(state, max_layers=layers) for layers in (1, 20)
            )
        ]
        assert fidelities[0] < fidelities[1]

    def test_layers_compose_the_circuit(self):
        """Test that the circuit applies the layers in order."""
        preparation = mps_state_preparation(random_state(4, seed=1), max_layers=3)
        product = Operator(np.eye(16))
        for layer in preparation.layers:
            product = Operator(layer) @ product
        assert_equal_operators(preparation.circuit, product)


class TestValidation:
    """Test the validation of the arguments."""

    def test_rejects_unnormalized_states(self):
        """Test that states must be normalized."""
        with pytest.raises(ValueError, match="normalized"):
            mps_state_preparation([1.0, 1.0], max_layers=1)

    def test_rejects_no_layers(self):
        """Test that at least one layer is allowed."""
        with pytest.raises(ValueError, match="at least 1"):
            mps_state_preparation([0.6, 0.8], max_layers=0)


def test_intensity_signal():
    """Test the preparation of the state of an intensity signal `f = alpha |psi|**2`."""
    intensity = Signal(PositionAxis(8, 1.0, IndexOrdering.NATURAL), np.arange(1.0, 9.0))
    alpha, state = sample_based_decomposition(intensity)
    preparation = mps_state_preparation(state, max_layers=1)

    prepared = Statevector(preparation.circuit)
    assert_close(alpha * np.abs(prepared.data) ** 2, intensity.data)
