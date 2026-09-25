"""Gaussian beams through paraxial optical systems, with ABCD ray transfer matrices.

A Gaussian beam of waist radius `w0`, a distance `d` behind its waist, is described by
its complex beam parameter `q = d + i z_R`, with the Rayleigh range
`z_R = pi w0**2 / wavelength`. A system of ray transfer matrix `[[A, B], [C, D]]` maps
it to `q' = (A q + B) / (C q + D)`, from which the radius of curvature `R` of the
wavefront and the beam radius `w` follow by `1 / q' = 1 / R - i wavelength / (pi w**2)`.

The matrices of a sequence of elements multiply in reverse order, the last element
first: `thin_lens_matrix(f)` followed by `propagation_matrix(d)` is
`propagation_matrix(d) @ thin_lens_matrix(f)`.
"""

import numpy as np
import numpy.typing as npt
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.physical_axis import PhysicalAxis


def propagation_matrix(distance: float) -> npt.NDArray[np.float64]:
    """Return the ray transfer matrix of a free propagation over a distance."""
    return np.array([[1.0, distance], [0.0, 1.0]])


def thin_lens_matrix(focal_length: float) -> npt.NDArray[np.float64]:
    """Return the ray transfer matrix of a thin lens of a focal length."""
    return np.array([[1.0, 0.0], [-1 / focal_length, 1.0]])


def rayleigh_range(waist: float, wavelength: float) -> float:
    """Return the Rayleigh range of a Gaussian beam of a waist radius."""
    return np.pi * waist**2 / wavelength


def beam_after_system(
    waist: float,
    distance_from_waist: float,
    wavelength: float,
    matrix: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Return the wavefront curvature radius and beam radius after an optical system.

    Args:
        waist: The waist radius of the incoming beam.
        distance_from_waist: The distance the beam propagated from its waist to the
            entrance of the system.
        wavelength: The wavelength in the medium.
        matrix: The ray transfer matrix of the system.

    Returns:
        The radius of curvature `R` of the wavefront, infinite for a flat one, and the
        beam radius `w`, at the exit of the system.
    """
    (a, b), (c, d) = matrix
    q_in = distance_from_waist + 1j * rayleigh_range(waist, wavelength)
    q_out = (a * q_in + b) / (c * q_in + d)
    inverse_q = 1 / q_out
    radius_of_curvature = np.inf if inverse_q.real == 0 else 1 / inverse_q.real
    beam_radius = np.sqrt(-wavelength / (np.pi * inverse_q.imag))
    return float(radius_of_curvature), float(beam_radius)


def beam_after_free_space(
    waist: float, distance: float, wavelength: float, refractive_index: float
) -> tuple[float, float]:
    """Return `(R, w)` of a beam after a propagation from its waist in a medium.

    Args:
        waist: The waist radius of the beam.
        distance: The propagation distance from the waist.
        wavelength: The vacuum wavelength.
        refractive_index: The refractive index of the medium.
    """
    return beam_after_system(
        waist, 0.0, wavelength / refractive_index, propagation_matrix(distance)
    )


def beam_after_thin_lens(
    waist: float,
    distance: float,
    focal_length: float,
    wavelength: float,
    refractive_index: float,
    distance_from_waist: float = 0.0,
) -> tuple[float, float]:
    """Return `(R, w)` of a beam after a thin lens and a propagation behind it.

    Args:
        waist: The waist radius of the incoming beam.
        distance: The propagation distance behind the lens.
        focal_length: The focal length of the lens.
        wavelength: The vacuum wavelength.
        refractive_index: The refractive index of the medium around the lens.
        distance_from_waist: The distance from the waist of the incoming beam to the
            lens.
    """
    matrix = propagation_matrix(distance) @ thin_lens_matrix(focal_length)
    return beam_after_system(
        waist, distance_from_waist, wavelength / refractive_index, matrix
    )


def gaussian_signal(axis: PhysicalAxis, waist: float, mean: float) -> AlgebraicSignal:
    """Return the field `exp(-(x - mean)**2 / waist**2)` of a Gaussian beam on an axis."""
    return AlgebraicSignal(axis, lambda x: np.exp(-((x - mean) ** 2) / waist**2))
