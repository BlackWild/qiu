"""Shared fixtures of the tests of qiu_classical_simulation.wave_optics."""

from collections.abc import Callable

import pytest
from qiu_classical_simulation.wave_optics.parameters import ExperimentParameters

SMALL_EXPERIMENT = {
    "vacuum_wavelength": 1e-6,
    "beam_FWHM": 25e-6,
    "focal_length": 200e-6,
    "refractive_index": 1.25,
    "propagation_after_lens": 300e-6,
    "transverse_length": 100e-6,
    "num_of_steps_after_lens": 4,
    "lens_slices": 12,
    "num_qubits": 5,
    "max_delta": 0.05,
    "lens_reverse_order": False,
    "fresnel_approximation": False,
    "scale_down_phases": True,
    "direct_propagator": True,
}
"""The experiment of the paper on a small grid, with few slices and steps."""


@pytest.fixture
def small_experiment() -> Callable[..., ExperimentParameters]:
    """Return a factory of small experiments, with parameters overridden."""
    return lambda **overrides: ExperimentParameters(**{**SMALL_EXPERIMENT, **overrides})
