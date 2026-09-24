"""Unit tests for sample_based_manual.py."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.physical_axis import AxisDomain
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiskit_phase_propagator.sample_based_manual import (
    phase_propagate_state,
    phase_propagate_state_with_arbitrary_signal,
    phase_propagate_state_with_constant_delta,
    phase_propagation_cycle,
)
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import (
    random_positive_signal,
    state_pairs_with_equal_qubits,
)
from qiskit_pytest_helper.propagation import exact_cycles, run_propagator

methods = st.sampled_from(list(SynthesisMethod))
deltas_lists = st.lists(st.floats(min_value=0.0, max_value=0.5), min_size=1, max_size=5)


class TestPhasePropagationCycle:
    """Test phase_propagation_cycle."""

    @given(
        states=state_pairs_with_equal_qubits(),
        delta=st.floats(min_value=-1.0, max_value=1.0),
        method=methods,
    )
    def test_exact_map(self, states, delta: float, method: SynthesisMethod):
        """Test the output state and the success probability of one cycle."""
        psi, phi = states
        output, probability = phase_propagation_cycle(
            psi, delta, PreparableState(phi, method)
        )

        unnormalized = psi.data * (1 + (np.exp(1j * delta) - 1) * np.abs(phi.data) ** 2)
        assert_equal_states(output, unnormalized / np.linalg.norm(unnormalized))
        assert np.isclose(probability, np.linalg.norm(unnormalized) ** 2)


class TestPhasePropagateState:
    """Test the propagation over several cycles."""

    @given(states=state_pairs_with_equal_qubits(), deltas=deltas_lists)
    def test_one_cycle_per_delta(self, states, deltas):
        """Test that every delta applies one cycle."""
        psi, phi = states
        output = phase_propagate_state(psi, deltas, PreparableState(phi))
        assert_equal_states(output, exact_cycles(psi.data, phi.data, deltas))

    @given(
        states=state_pairs_with_equal_qubits(),
        delta=st.floats(min_value=0.0, max_value=0.5),
        num_cycles=st.integers(min_value=1, max_value=6),
    )
    def test_constant_delta_compounds(self, states, delta: float, num_cycles: int):
        """Test that all cycles are applied, each to the output of the previous one."""
        psi, phi = states
        output = phase_propagate_state_with_constant_delta(
            psi, delta, num_cycles, PreparableState(phi)
        )
        assert_equal_states(
            output, exact_cycles(psi.data, phi.data, np.full(num_cycles, delta))
        )

    @settings(max_examples=10, deadline=None)
    @given(
        signal=random_positive_signal(
            domain=AxisDomain.POSITION, max_qubits=3, forced_sum_value=0.1
        ),
        max_delta=st.floats(min_value=0.01, max_value=0.05),
    )
    def test_agrees_with_the_circuit(self, signal: AlgebraicSignal, max_delta: float):
        """Test that the simulation agrees with the circuit on the Aer simulator."""
        alpha, _ = sample_based_decomposition(signal)
        num_qubits = signal.axis.size.bit_length() - 1
        psi = Statevector.from_label("+" * num_qubits)

        # Qiskit fails to transpile its StatePreparation of nearly uniform states
        method = SynthesisMethod.DECOMPOSED
        simulated = phase_propagate_state_with_arbitrary_signal(
            psi, signal, max_delta, method
        )
        succeeded, output = run_propagator(
            QuadraticSignalSampleBasedPhasePropagator(signal, max_delta, method), psi
        )

        assert len(slice_alpha_to_deltas_evenly(alpha, max_delta)) >= 1
        if succeeded:
            assert_equal_states(output, simulated)
