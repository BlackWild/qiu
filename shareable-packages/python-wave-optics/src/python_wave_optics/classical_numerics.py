"""Classical references of a lens experiment: split-step numerics and thin lens analytics."""

import numpy as np
import numpy.typing as npt

from python_wave_optics.elements import free_space_propagator_phase
from python_wave_optics.gaussian_beam import beam_after_thin_lens, gaussian_signal
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.simulation import (
    from_angular_spectrum,
    has_phase,
    to_angular_spectrum,
)


def propagate_exactly(
    state: npt.NDArray[np.complex128], parameters: ExperimentParameters, distance: float
) -> npt.NDArray[np.complex128]:
    """Return a field after a free propagation over a distance in vacuum."""
    signal = free_space_propagator_phase(
        parameters.k_axis, distance, parameters.vacuum_wavelength
    )
    return from_angular_spectrum(np.exp(1j * signal.data) * to_angular_spectrum(state))


def classical_numerics_simulation(
    parameters: ExperimentParameters, propagation_after_lens: float
) -> npt.NDArray[np.complex128]:
    """Return the exact split-step field at a distance behind the lens.

    The same slicing of the lens as the simulations, with the phases and propagations
    applied exactly, and a single propagation behind the lens.

    Args:
        parameters: The experiment.
        propagation_after_lens: The distance behind the lens.

    Returns:
        The normalized field.
    """
    state = parameters.initial_state
    for signal in parameters.ordered_lens_signals:
        if has_phase(signal):
            state = state * np.exp(1j * signal.data)
        state = propagate_exactly(state, parameters, parameters.lens_slice_thickness)
    return propagate_exactly(state, parameters, propagation_after_lens)


def thin_lens_simulation(
    parameters: ExperimentParameters,
    propagation_before_lens: float,
    propagation_after_lens: float,
) -> npt.NDArray[np.float64]:
    """Return the analytic Gaussian profile behind an ideal thin lens of the same focus.

    Args:
        parameters: The experiment.
        propagation_before_lens: The distance from the beam's waist to the thin lens.
        propagation_after_lens: The distance behind the thin lens.

    Returns:
        The normalized magnitude of the field.
    """
    _, beam_radius = beam_after_thin_lens(
        parameters.gaussian_beam_waist,
        propagation_after_lens,
        parameters.focal_length,
        parameters.vacuum_wavelength,
        refractive_index=1,
        distance_from_waist=propagation_before_lens,
    )
    profile = gaussian_signal(parameters.x_axis, beam_radius, parameters.gaussian_mean)
    return np.asarray(profile.to_signal().normalized_data, dtype=np.float64)
