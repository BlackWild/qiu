"""Unit tests for time_independent/direct.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.hypothesis_strategies import monomial_signals
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain, MomentumAxis, PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
    PositionDomainEvolutionQuadratic,
)
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import (
    moderate_alphas,
    qubit_axes,
)


def random_state(dimension: int) -> np.ndarray:
    """A random normalized state, generic in position and momentum space."""
    rng = np.random.default_rng(dimension)
    data = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return data / np.linalg.norm(data)


def quadratic(signal) -> QuadraticSignal:
    """Convert a monomial of power 2 to a quadratic signal."""
    return QuadraticSignal(signal.axis, signal.alpha)


class TestPositionDomainEvolutionQuadratic:
    """Unit tests for the PositionDomainEvolutionQuadratic circuit."""

    @given(
        signal=monomial_signals(
            qubit_axes(AxisDomain.POSITION), alphas=moderate_alphas, powers=st.just(2)
        )
    )
    def test_applies_the_phase(self, signal):
        """Test that the position amplitudes are multiplied by e^(i f(x))."""
        signal = quadratic(signal)
        psi = random_state(signal.axis.size)

        output = Statevector(psi).evolve(PositionDomainEvolutionQuadratic(signal))
        assert_equal_states(output, np.exp(1j * signal.data) * psi)

    def test_rejects_momentum_axes(self):
        """Test that the signal must live in the position domain."""
        signal = QuadraticSignal(MomentumAxis(4, 0.5, IndexOrdering.FFT), 0.1)
        with pytest.raises(ValueError, match="position domain"):
            PositionDomainEvolutionQuadratic(signal)


class TestMomentumDomainEvolutionQuadratic:
    """Unit tests for the MomentumDomainEvolutionQuadratic circuit."""

    @given(
        signal=monomial_signals(
            qubit_axes(AxisDomain.MOMENTUM, orderings=st.just(IndexOrdering.FFT)),
            alphas=moderate_alphas,
            powers=st.just(2),
        )
    )
    def test_applies_the_phase_in_momentum_space(self, signal):
        """Test that the momentum amplitudes, fft(psi), are multiplied by e^(i f(p))."""
        signal = quadratic(signal)
        psi = random_state(signal.axis.size)

        for method in SynthesisMethod:
            output = Statevector(psi).evolve(
                MomentumDomainEvolutionQuadratic(signal, fourier_method=method)
            )
            expected = np.fft.ifft(
                np.exp(1j * signal.data) * np.fft.fft(psi, norm="ortho"), norm="ortho"
            )
            assert_equal_states(output, expected)

    def test_rejects_position_axes(self):
        """Test that the signal must live in a Fourier domain."""
        signal = QuadraticSignal(PositionAxis(4, 0.5, IndexOrdering.FFT), 0.1)
        with pytest.raises(ValueError, match="Fourier domain"):
            MomentumDomainEvolutionQuadratic(signal)

    @pytest.mark.parametrize(
        "ordering", [IndexOrdering.NATURAL, IndexOrdering.CENTERED]
    )
    def test_rejects_other_orderings(self, ordering: IndexOrdering):
        """Test that only the FFT ordering matches the Fourier basis states."""
        signal = QuadraticSignal(MomentumAxis(4, 0.5, ordering), 0.1)
        with pytest.raises(ValueError, match="FFT ordering"):
            MomentumDomainEvolutionQuadratic(signal)
