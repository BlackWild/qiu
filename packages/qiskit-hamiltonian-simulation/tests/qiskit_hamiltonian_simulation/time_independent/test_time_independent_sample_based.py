"""Unit tests for time_independent/sample_based.py."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain, MomentumAxis, PositionAxis
from python_signals.signal import Signal
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_hamiltonian_simulation.time_independent.sample_based import (
    KineticEvolutionSampleBased,
    PotentialEvolutionSampleBased,
)
from qiskit_phase_propagator.sample_based import (
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.constants import (
    MAX_NUM_OF_CYCLES,
    REDUCED_FIDELITY_TOLERANCE,
)
from qiskit_pytest_helper.hypothesis_strategies import random_positive_signal
from qiskit_pytest_helper.propagation import exact_cycles, run_propagator

HBAR = 1.0
times = st.floats(0.1, 0.3)
max_deltas = st.floats(min_value=0.01, max_value=0.1)
# Qiskit fails to transpile its StatePreparation of nearly uniform states (qiskit 2.2)
STATE_PREPARATION_METHOD = SynthesisMethod.DECOMPOSED


def exact_phase_propagation(signal, max_delta: float, psi) -> np.ndarray:
    """The exact output of the sample-based propagator for the signal on psi."""
    alpha, state = sample_based_decomposition(signal)
    deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)
    return exact_cycles(psi, state.data, deltas)


def random_state(num_qubits: int, seed: int) -> Statevector:
    """A random normalized state, generic in position and momentum space."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    return Statevector(data / np.linalg.norm(data))


class TestPotentialEvolutionSampleBased:
    """Tests for the PotentialEvolutionSampleBased circuit."""

    @settings(max_examples=50, deadline=None)
    @given(
        V=random_positive_signal(domain=AxisDomain.POSITION),
        t=times,
        max_delta=max_deltas,
    )
    def test_reasonable_num_of_iterations(self, V, t: float, max_delta: float):
        """Tests that the number of cycles is reasonable."""
        propagator = PotentialEvolutionSampleBased(
            V=V, t=t, hbar=HBAR, max_delta=max_delta
        )
        assert propagator.num_of_cycles < MAX_NUM_OF_CYCLES

    @settings(max_examples=10, deadline=None)
    @given(
        V=random_positive_signal(domain=AxisDomain.POSITION),
        t=times,
        max_delta=max_deltas,
    )
    def test_applies_the_evolution(self, V, t: float, max_delta: float):
        """Tests that the position amplitudes are multiplied by e^(-i t V / hbar)."""
        propagator = PotentialEvolutionSampleBased(
            V=V,
            t=t,
            hbar=HBAR,
            max_delta=max_delta,
            state_preparation_method=STATE_PREPARATION_METHOD,
        )
        psi = random_state(propagator.num_qubits // 2, seed=1)

        succeeded, output = run_propagator(propagator, psi)
        assume(succeeded)

        assert_equal_states(
            output, exact_phase_propagation((-t / HBAR) * V, max_delta, psi.data)
        )
        expected = np.exp(-1j * t / HBAR * V.data) * psi.data
        assert (
            state_fidelity(Statevector(output), Statevector(expected))
            >= 1 - REDUCED_FIDELITY_TOLERANCE
        )

    def test_accepts_sampled_signals(self):
        """Tests that sampled potentials are accepted."""
        V = Signal(PositionAxis(4, 0.5, IndexOrdering.NATURAL), [0.1, 0.2, 0.3, 0.4])
        propagator = PotentialEvolutionSampleBased(
            V=V, t=0.5, hbar=HBAR, max_delta=0.05
        )
        assert propagator.num_of_cycles == 10

    def test_rejects_momentum_axes(self):
        """Tests that the potential must live in the position domain."""
        V = AlgebraicSignal(MomentumAxis(4, 0.5, IndexOrdering.FFT), lambda p: 1 + p**2)
        with pytest.raises(ValueError, match="position domain"):
            PotentialEvolutionSampleBased(V=V, t=0.1, hbar=HBAR, max_delta=0.1)


class TestKineticEvolutionSampleBased:
    """Tests for the KineticEvolutionSampleBased circuit."""

    @settings(max_examples=50, deadline=None)
    @given(
        T=random_positive_signal(
            domain=AxisDomain.MOMENTUM, forced_ordering=IndexOrdering.FFT
        ),
        t=times,
        max_delta=max_deltas,
    )
    def test_reasonable_num_of_iterations(self, T, t: float, max_delta: float):
        """Tests that the number of cycles is reasonable."""
        propagator = KineticEvolutionSampleBased(
            T=T, t=t, hbar=HBAR, max_delta=max_delta
        )
        assert propagator.num_of_cycles < MAX_NUM_OF_CYCLES

    @settings(max_examples=10, deadline=None)
    @given(
        T=random_positive_signal(
            domain=AxisDomain.MOMENTUM, forced_ordering=IndexOrdering.FFT
        ),
        t=times,
        max_delta=max_deltas,
    )
    def test_applies_the_evolution_in_momentum_space(
        self, T, t: float, max_delta: float
    ):
        """Tests that the momentum amplitudes, fft(psi), get e^(-i t T(p) / hbar).

        The random polynomial kinetic energies are not symmetric in p, which checks
        the direction of the Fourier transforms.
        """
        propagator = KineticEvolutionSampleBased(
            T=T,
            t=t,
            hbar=HBAR,
            max_delta=max_delta,
            state_preparation_method=STATE_PREPARATION_METHOD,
        )
        psi = random_state(propagator.num_qubits // 2, seed=2)

        succeeded, output = run_propagator(propagator, psi)
        assume(succeeded)

        momentum_amplitudes = np.fft.fft(psi.data, norm="ortho")
        assert_equal_states(
            output,
            np.fft.ifft(
                exact_phase_propagation(
                    (-t / HBAR) * T, max_delta, momentum_amplitudes
                ),
                norm="ortho",
            ),
        )
        expected = np.fft.ifft(
            np.exp(-1j * t / HBAR * T.data) * momentum_amplitudes, norm="ortho"
        )
        assert (
            state_fidelity(Statevector(output), Statevector(expected))
            >= 1 - REDUCED_FIDELITY_TOLERANCE
        )

    @pytest.mark.parametrize(
        "ordering", [IndexOrdering.NATURAL, IndexOrdering.CENTERED]
    )
    def test_rejects_other_orderings(self, ordering: IndexOrdering):
        """Tests that only the FFT ordering matches the Fourier basis states."""
        T = AlgebraicSignal(MomentumAxis(4, 0.5, ordering), lambda p: 1 + p**2)
        with pytest.raises(ValueError, match="FFT ordering"):
            KineticEvolutionSampleBased(T=T, t=0.1, hbar=HBAR, max_delta=0.1)
