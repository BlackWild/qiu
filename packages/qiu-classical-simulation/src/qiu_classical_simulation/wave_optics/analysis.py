"""Quantities to analyze a lens experiment with, e.g. its beam waist along the way."""

import numpy as np
import numpy.typing as npt

from qiu_classical_simulation.wave_optics.classical_numerics import thin_lens_simulation
from qiu_classical_simulation.wave_optics.parameters import ExperimentParameters


def beam_waist(state: npt.ArrayLike, x_values: npt.ArrayLike) -> float:
    """Return the beam waist of a field, twice the standard deviation of its intensity.

    This is the waist radius of a Gaussian beam `exp(-x**2 / w**2)`.
    """
    intensity = np.abs(np.asarray(state)) ** 2
    x = np.asarray(x_values)
    mean = np.sum(x * intensity) / np.sum(intensity)
    variance = np.sum(x**2 * intensity) / np.sum(intensity) - mean**2
    return float(2 * np.sqrt(variance))


def principal_plane_position(parameters: ExperimentParameters) -> float:
    """Return the depth of the principal plane of the lens, from where the beam enters.

    It is at the vertex of the convex surface, which is the exit for the reverse order;
    for the forward order, it lies at `t - t / n` for the thickness `t`.
    """
    if parameters.lens_reverse_order:
        return parameters.lens_thickness
    return parameters.lens_thickness - parameters.lens_thickness / (
        parameters.refractive_index
    )


def propagation_distances(parameters: ExperimentParameters) -> list[float]:
    """Return the propagation distances of the snapshots after each slice and step."""
    return [
        parameters.lens_slice_thickness * (i + 1) for i in range(parameters.lens_slices)
    ] + [
        parameters.lens_thickness + parameters.step_size_after_lens * (j + 1)
        for j in range(parameters.num_of_steps_after_lens)
    ]


def thin_lens_reference_states(
    parameters: ExperimentParameters,
) -> list[npt.NDArray[np.float64]]:
    """Return the thin lens profiles at the distances of `propagation_distances`.

    The ideal thin lens sits at the principal plane of the lens.
    """
    principal_plane = principal_plane_position(parameters)
    return [
        thin_lens_simulation(parameters, distance, 0)
        if distance < principal_plane
        else thin_lens_simulation(
            parameters, principal_plane, distance - principal_plane
        )
        for distance in propagation_distances(parameters)
    ]


def lens_surface(parameters: ExperimentParameters) -> npt.NDArray[np.float64]:
    """Return the depth of the convex lens surface at each transverse position.

    The depth is measured from where the beam enters, as in `propagation_distances`.
    """
    x = np.asarray(parameters.x_axis.values, dtype=np.float64)
    x = x - parameters.transverse_length / 2
    radius = parameters.radius_of_curvature
    if parameters.fresnel_approximation:
        sag = x**2 / (2 * radius)
    else:
        sag = radius - np.sqrt(radius**2 - x**2)
    return parameters.lens_thickness - sag if parameters.lens_reverse_order else sag
