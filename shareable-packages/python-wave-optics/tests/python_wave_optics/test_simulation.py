"""Unit tests for simulation.py, classical_numerics.py and phase_protocol.py."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import RTOL, assert_close, assert_close_in_norm
from python_pytest_helper.hypothesis_strategies import (
    physical_axes,
    positive_polynomial_signals,
    power_of_two_sizes,
)
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal
from python_wave_optics.classical_numerics import (
    classical_numerics_simulation,
    propagate_exactly,
)
from python_wave_optics.phase_protocol import decompose, ideal_cycles, slice_phase
from python_wave_optics.simulation import (
    ExactBackend,
    free_propagation,
    has_phase,
    simulate,
)

AXIS = PositionAxis(4, 1.0, "natural")  # type: ignore[arg-type]


class TestPhaseProtocol:
    """Test the arithmetic of the phase protocol."""

    @given(signal=positive_polynomial_signals(total=0.3))
    def test_decompose(self, signal: AlgebraicSignal):
        """Test `f = alpha |phi|**2` with a normalized `phi`."""
        alpha, phi = decompose(signal)
        assert_close(alpha, 0.3)
        assert_close(np.linalg.norm(phi), 1.0)
        assert_close(alpha * phi**2, signal.data)

    def test_decompose_non_positive(self):
        """Test that non-positive signals have a negative alpha."""
        alpha, phi = decompose(Signal(AXIS, [-1.0, -3.0, 0.0, -4.0]))
        assert alpha == -8.0
        assert_close(phi**2, [1 / 8, 3 / 8, 0, 4 / 8])

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ([1.0, -1.0, 1, 1], "same sign"),
            ([0.0] * 4, "vanish"),
            ([1j, 1, 1, 1], "real"),
        ],
    )
    def test_decompose_invalid(self, data, match: str):
        """Test that mixed signs, vanishing and complex signals are rejected."""
        with pytest.raises(ValueError, match=match):
            decompose(Signal(AXIS, data))

    @given(
        alpha=st.floats(min_value=-10, max_value=10),
        max_delta=st.floats(min_value=0.01, max_value=1),
    )
    def test_slice_phase(self, alpha: float, max_delta: float):
        """Test the fewest equal phases of magnitude at most max_delta summing to alpha."""
        deltas = slice_phase(alpha, max_delta)
        assert len(deltas) == int(np.ceil(abs(alpha) / max_delta))
        if len(deltas):
            assert_close(np.sum(deltas), alpha)
            assert np.all(np.abs(deltas) <= max_delta * (1 + RTOL))

    @given(
        signal=positive_polynomial_signals(
            physical_axes(sizes=power_of_two_sizes(1, 4)), total=0.2
        )
    )
    def test_ideal_cycles_apply_the_phase(self, signal: AlgebraicSignal):
        """Test that the cycles apply `e^(i f)` up to `O(delta**2)`."""
        alpha, phi = decompose(signal)
        psi = np.full(signal.axis.size, 1 / np.sqrt(signal.axis.size), dtype=complex)
        state, probability = ideal_cycles(psi, phi, slice_phase(alpha, 0.01))
        exact = np.exp(1j * signal.data) * psi
        # the cycles err by O(delta**2) each, i.e. by O(alpha delta) = 2e-3 in total
        assert abs(np.vdot(state, exact)) ** 2 > 1 - 1e-4
        assert 0 < probability <= 1


class TestSimulate:
    """Test simulate with the exact backend."""

    def test_snapshots(self, small_experiment):
        """Test the order of the snapshots and their success probabilities."""
        params = small_experiment()
        result = simulate(params, ExactBackend())
        expected = (
            ["step_0"]
            + [f"step_lens_{i}" for i in range(params.lens_slices)]
            + ["after_lens"]
            + [
                f"step_after_lens_{j + 1}"
                for j in range(params.num_of_steps_after_lens)
            ]
            + ["final"]
        )
        assert list(result.snapshots) == expected
        assert result.success_probabilities == [1.0] * len(expected)
        assert result.total_lenses_simulated == params.lens_slices

    @pytest.mark.parametrize("direct_propagator", [True, False])
    @pytest.mark.parametrize("lens_reverse_order", [False, True])
    def test_agrees_with_the_classical_numerics(
        self, small_experiment, direct_propagator: bool, lens_reverse_order: bool
    ):
        """Test that the exact simulation is the split-step classical numerics."""
        params = small_experiment(
            direct_propagator=direct_propagator, lens_reverse_order=lens_reverse_order
        )
        result = simulate(params, ExactBackend())
        for step in range(1, params.num_of_steps_after_lens + 1):
            assert_close_in_norm(
                result.snapshots[f"step_after_lens_{step}"],
                classical_numerics_simulation(
                    params, params.step_size_after_lens * step
                ),
            )

    def test_progress(self, small_experiment):
        """Test that the progress wraps the loops over slices and steps."""
        names = []
        simulate(
            small_experiment(),
            ExactBackend(),
            progress=lambda loop, name: names.append(name) or loop,
        )
        assert names == ["Lens slices", "Free space steps"]


class TestFreePropagation:
    """Test free_propagation and propagate_exactly."""

    def test_distances_add_up(self, small_experiment):
        """Test that two propagations are one over the sum of their distances."""
        params = small_experiment()
        state = params.initial_state
        twice = propagate_exactly(propagate_exactly(state, params, 1e-5), params, 2e-5)
        assert_close_in_norm(twice, propagate_exactly(state, params, 3e-5))

    def test_sample_based_on_the_angular_spectrum(self, small_experiment):
        """Test that the sample-based propagation acts on the angular spectrum."""
        params = small_experiment(direct_propagator=False)
        state, probability = free_propagation(ExactBackend(), params, 1e-5)(
            params.initial_state
        )
        assert probability == 1.0
        assert_close_in_norm(
            state, propagate_exactly(params.initial_state, params, 1e-5)
        )


def test_has_phase():
    """Test that constant phases, i.e. global ones, are skipped."""
    assert not has_phase(Signal(AXIS, [0.3, 0.3, 0.3, 0.3]))
    assert has_phase(Signal(AXIS, [0.0, 0.3, 0.3, 0.0]))
