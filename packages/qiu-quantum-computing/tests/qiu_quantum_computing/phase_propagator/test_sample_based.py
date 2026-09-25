"""Unit tests for sample_based.py."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from python_pytest_helper.assertions import RTOL, assert_close
from python_pytest_helper.hypothesis_strategies import positive_polynomial_signals
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_pytest_helper.assertions import (
    assert_equal_operators,
    assert_equal_states,
)
from qiskit_pytest_helper.constants import FIDELITY_TOLERANCE
from qiskit_pytest_helper.hypothesis_strategies import (
    qubit_axes,
    state_pairs_with_equal_qubits,
)
from qiskit_pytest_helper.propagation import exact_cycles, run_propagator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    GenericIterativeSampleBasedPhasePropagator,
    GenericIterativeSampleBasedPhasePropagatorWithConstantDelta,
    QuadraticSignalSampleBasedPhasePropagator,
    partial_phase_circuit,
    partial_phase_diagonal,
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import AxisDomain, PositionAxis
from qiu_signals.signal import Signal

AXIS = PositionAxis(4, 0.5, IndexOrdering.FFT)
deltas_lists = st.lists(st.floats(min_value=0.0, max_value=0.1), min_size=1, max_size=4)


class TestSampleBasedDecomposition:
    """Test sample_based_decomposition."""

    @given(signal=positive_polynomial_signals(qubit_axes(AxisDomain.POSITION)))
    def test_reconstructs_the_signal(self, signal: AlgebraicSignal):
        """Test that the signal is alpha times the squared amplitudes of the state."""
        alpha, state = sample_based_decomposition(signal)
        assert state.is_valid()
        assert_close(alpha * np.abs(state.data) ** 2, signal.data)

    def test_non_positive_signals(self):
        """Test that non-positive signals have a negative alpha."""
        alpha, state = sample_based_decomposition(Signal(AXIS, [-1.0, -3.0, 0.0, -4.0]))
        assert alpha == -8.0
        assert_close(np.abs(state.data) ** 2, [1 / 8, 3 / 8, 0, 4 / 8])

    def test_accepts_real_complex_data(self):
        """Test that complex data with vanishing imaginary parts is accepted."""
        alpha, _ = sample_based_decomposition(Signal(AXIS, np.ones(4, dtype=complex)))
        assert alpha == 4.0

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ([1.0, -1.0, 1.0, 1.0], "same sign"),
            ([0.0, 0.0, 0.0, 0.0], "vanish"),
            ([1j, 1.0, 1.0, 1.0], "real"),
        ],
    )
    def test_invalid_signals(self, data, match: str):
        """Test that mixed signs, vanishing and complex signals are rejected."""
        with pytest.raises(ValueError, match=match):
            sample_based_decomposition(Signal(AXIS, data))

    def test_axes_must_have_qubit_sizes(self):
        """Test that the axis must have 2**n samples."""
        signal = Signal(PositionAxis(3, 1.0, IndexOrdering.FFT), np.ones(3))
        with pytest.raises(ValueError, match="2\\*\\*n"):
            sample_based_decomposition(signal)


class TestSliceAlphaToDeltasEvenly:
    """Test slice_alpha_to_deltas_evenly."""

    @given(
        alpha=st.floats(min_value=-10, max_value=10),
        max_delta=st.floats(min_value=0.01, max_value=1),
    )
    def test_slices(self, alpha: float, max_delta: float):
        """Test that the fewest equal deltas of bounded magnitude sum up to alpha."""
        deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)

        assert_close(np.sum(deltas), alpha)
        assert np.all(np.abs(deltas) <= max_delta * (1 + RTOL))
        assert len(deltas) == int(np.ceil(abs(alpha) / max_delta))
        assert np.all(deltas == deltas[0]) if len(deltas) else alpha == 0

    def test_invalid_max_delta(self):
        """Test that max_delta must be positive."""
        with pytest.raises(ValueError, match="positive"):
            slice_alpha_to_deltas_evenly(1.0, 0.0)


class TestPartialPhaseCircuit:
    """Test partial_phase_circuit."""

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    def test_phase_where_the_registers_agree(self, num_qubits: int):
        """Test that |j>|l> gets the phase e^(i delta) exactly if j == l."""
        delta = 0.37
        dimension = 2**num_qubits
        psi_index = np.arange(dimension**2) % dimension
        phi_index = np.arange(dimension**2) // dimension

        assert_equal_operators(
            partial_phase_circuit(delta, num_qubits),
            np.diag(np.exp(1j * delta * (psi_index == phi_index))),
        )

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    def test_diagonal(self, num_qubits: int):
        """Test that partial_phase_diagonal is the diagonal of the circuit."""
        assert_equal_operators(
            partial_phase_circuit(0.37, num_qubits),
            np.diag(partial_phase_diagonal(0.37, num_qubits)),
        )


class TestGenericIterativeSampleBasedPhasePropagator:
    """Test the GenericIterativeSampleBasedPhasePropagator."""

    @given(states=state_pairs_with_equal_qubits(), deltas=deltas_lists)
    def test_essentials(self, states: tuple[Statevector, Statevector], deltas):
        """Test the registers and the number of cycles."""
        _, phi = states
        propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
            PreparableState(phi, method=SynthesisMethod.DENSE), deltas
        )
        assert propagator.num_qubits == 2 * phi.num_qubits  # type: ignore[operator]
        assert propagator.num_clbits == phi.num_qubits
        assert propagator.num_of_cycles == len(deltas)

    @settings(max_examples=10, deadline=None)
    @given(states=state_pairs_with_equal_qubits(), deltas=deltas_lists)
    def test_applies_the_cycles(self, states: tuple[Statevector, Statevector], deltas):
        """Test that each successful cycle applies its exact map to psi."""
        psi, phi = states
        propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
            PreparableState(phi, method=SynthesisMethod.DENSE), deltas
        )

        succeeded, output = run_propagator(propagator, psi)
        assume(succeeded)
        assert_equal_states(output, exact_cycles(psi.data, phi.data, deltas))


class TestGenericIterativeSampleBasedPhasePropagatorWithConstantDelta:
    """Test the GenericIterativeSampleBasedPhasePropagatorWithConstantDelta."""

    @settings(max_examples=10, deadline=None)
    @given(
        states=state_pairs_with_equal_qubits(),
        delta=st.floats(min_value=0.0, max_value=0.1),
        number_of_cycles=st.integers(min_value=1, max_value=10),
    )
    def test_applies_the_cycles(
        self,
        states: tuple[Statevector, Statevector],
        delta: float,
        number_of_cycles: int,
    ):
        """Test that the loop applies the exact map of each cycle."""
        psi, phi = states
        propagator = (
            GenericIterativeSampleBasedPhasePropagatorWithConstantDelta.from_state(
                PreparableState(phi, method=SynthesisMethod.DENSE),
                delta,
                number_of_cycles,
            )
        )
        assert propagator.num_of_cycles == number_of_cycles

        succeeded, output = run_propagator(propagator, psi)
        assume(succeeded)
        assert_equal_states(
            output, exact_cycles(psi.data, phi.data, np.full(number_of_cycles, delta))
        )


class TestQuadraticSignalSampleBasedPhasePropagator:
    """Test the QuadraticSignalSampleBasedPhasePropagator."""

    @settings(max_examples=10, deadline=None)
    @given(
        signal=positive_polynomial_signals(
            qubit_axes(AxisDomain.POSITION, max_qubits=3)
        ),
        max_delta=st.floats(min_value=0.01, max_value=0.1),
    )
    def test_essentials(self, signal: AlgebraicSignal, max_delta: float):
        """Test the registers and the number of cycles."""
        propagator = QuadraticSignalSampleBasedPhasePropagator(signal, max_delta)
        alpha, _ = sample_based_decomposition(signal)

        assert propagator.num_qubits == 2 * signal.axis.size.bit_length() - 2
        assert propagator.num_of_cycles == len(
            slice_alpha_to_deltas_evenly(alpha, max_delta)
        )

    @settings(max_examples=10, deadline=None)
    @given(
        signal=positive_polynomial_signals(
            qubit_axes(AxisDomain.POSITION, max_qubits=3), total=0.1
        ),
        max_delta=st.floats(min_value=0.01, max_value=0.05),
        method=st.sampled_from([SynthesisMethod.DENSE, SynthesisMethod.DECOMPOSED]),
    )
    def test_applies_the_signal(
        self, signal: AlgebraicSignal, max_delta: float, method: SynthesisMethod
    ):
        """Test that the propagator applies e^(i f) up to the slicing error."""
        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal, max_delta, method
        )
        psi = Statevector.from_label("+" * (propagator.num_qubits // 2))

        succeeded, output = run_propagator(propagator, psi)
        assume(succeeded)

        alpha, state = sample_based_decomposition(signal)
        deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)
        assert_equal_states(output, exact_cycles(psi.data, state.data, deltas))
        assert (
            state_fidelity(
                Statevector(output), Statevector(np.exp(1j * signal.data) * psi.data)
            )
            >= 1 - FIDELITY_TOLERANCE
        )

    def test_accepts_sampled_signals(self):
        """Test that sampled signals are accepted as well as algebraic ones."""
        signal = Signal(AXIS, [0.1, 0.2, 0.3, 0.4])
        propagator = QuadraticSignalSampleBasedPhasePropagator(signal, max_delta=0.1)
        assert propagator.num_of_cycles == 10
