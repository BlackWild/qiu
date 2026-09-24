"""Unit tests for the QuTiP backend."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close, assert_close_in_norm
from python_pytest_helper.hypothesis_strategies import (
    monomial_signals,
    physical_axes,
    positive_polynomial_signals,
    power_of_two_sizes,
)
from python_signals.algebraic_signal import AlgebraicSignal, PolynomialSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.phase_protocol import decompose, ideal_cycles, slice_phase
from python_wave_optics.simulation import ExactBackend, simulate
from wave_optics_propagation_qutip.backend import (
    QutipBackend,
    dft_operator,
    partial_phase_operator,
)

max_deltas = st.floats(min_value=0.01, max_value=0.2)


def random_state(size: int) -> np.ndarray:
    """A random normalized state."""
    rng = np.random.default_rng(size)
    data = rng.normal(size=size) + 1j * rng.normal(size=size)
    return data / np.linalg.norm(data)


class TestOperators:
    """Test the operators of the backend."""

    @pytest.mark.parametrize("dimension", [2, 4, 8])
    def test_partial_phase(self, dimension: int):
        """Test the phase exactly where the ancilla and the field agree."""
        diagonal = partial_phase_operator(0.3, dimension).full().diagonal()
        ancilla, field = np.divmod(np.arange(dimension**2), dimension)
        assert_close(diagonal, np.exp(0.3j * (ancilla == field)))

    @pytest.mark.parametrize("dimension", [2, 4, 8])
    def test_dft(self, dimension: int):
        """Test that the DFT is numpy's orthonormal FFT."""
        psi = random_state(dimension)
        applied = dft_operator(dimension).full() @ psi
        assert_close_in_norm(applied, np.fft.fft(psi, norm="ortho"))


class TestSampleBasedPhase:
    """Test the phase protocol of the QuTiP backend."""

    @settings(deadline=None)
    @given(
        signal=positive_polynomial_signals(
            physical_axes(sizes=power_of_two_sizes(1, 4)), total=0.5
        ),
        max_delta=max_deltas,
    )
    def test_is_the_ideal_protocol(self, signal: AlgebraicSignal, max_delta: float):
        """Test the state and the success probability of the post-selected cycles."""
        psi = random_state(signal.axis.size)
        state, probability = QutipBackend().sample_based_phase(signal, max_delta)(psi)

        alpha, phi = decompose(signal)
        expected, expected_probability = ideal_cycles(
            psi, phi, slice_phase(alpha, max_delta)
        )
        assert_close_in_norm(state, expected)
        assert_close(probability, expected_probability)


class TestDirectMomentumPhase:
    """Test the direct propagator of the QuTiP backend."""

    @given(
        signal=monomial_signals(
            physical_axes(
                AxisDomain.ANGULAR_WAVENUMBER,
                sizes=power_of_two_sizes(1, 4),
                orderings=st.just(IndexOrdering.FFT),
            ),
            powers=st.just(2),
        )
    )
    def test_is_the_exact_propagator(self, signal: PolynomialSignal):
        """Test that the angular spectrum gets the phase exactly."""
        psi = random_state(signal.axis.size)
        state, probability = QutipBackend().direct_momentum_phase(signal)(psi)  # type: ignore[arg-type]
        expected, _ = ExactBackend().direct_momentum_phase(signal)(psi)  # type: ignore[arg-type]
        assert_close_in_norm(state, expected)
        assert probability == 1.0


@pytest.mark.parametrize("direct_propagator", [True, False])
def test_simulation_approaches_the_exact_one(direct_propagator: bool):
    """Test that the simulated experiment approaches the exact one for small deltas."""
    params = ExperimentParameters(
        vacuum_wavelength=1e-6,
        beam_FWHM=25e-6,
        focal_length=200e-6,
        refractive_index=1.25,
        propagation_after_lens=300e-6,
        transverse_length=100e-6,
        num_of_steps_after_lens=3,
        lens_slices=8,
        num_qubits=4,
        max_delta=0.005,
        lens_reverse_order=True,
        fresnel_approximation=False,
        scale_down_phases=True,
        direct_propagator=direct_propagator,
    )
    simulated = simulate(params, QutipBackend())
    exact = simulate(params, ExactBackend())
    fidelity = abs(np.vdot(simulated.snapshots["final"], exact.snapshots["final"])) ** 2
    # the protocol errs by O(max_delta) in the phase of each slice and step
    assert fidelity > 0.999
    probability = simulated.total_probability_of_success
    assert probability is not None and 0 < probability <= 1
