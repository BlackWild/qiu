"""Phase signals of optical elements in the paraxial, thin element approximation.

A transverse field `psi(x)` passing a thin element is multiplied by `e^(i f(x))` with the
element's phase signal `f`, on a position axis. A free propagation over a distance `dz`
multiplies its angular spectrum `psi(k)` by `e^(i f(k))`, with the paraxial (Fresnel)
phase `f(k) = -k**2 dz / (2 k0)` on an angular wavenumber axis, `k0` being the vacuum
wavenumber `2 pi / wavelength`.
"""

import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from python_signals.physical_axis import AngularWavenumberAxis, PositionAxis


def vacuum_wavenumber(wavelength: float) -> float:
    """Return the wavenumber `2 pi / wavelength`."""
    return 2 * np.pi / wavelength


def convex_planar_lens_radius(
    radius_of_curvature: float,
    depth: float,
    lens_thickness: float,
    fresnel_approximation: bool,
) -> float:
    """Return the transverse radius of a plano-convex lens at a depth from its vertex.

    Args:
        radius_of_curvature: The radius of curvature of the convex surface.
        depth: The distance from the vertex along the optical axis.
        lens_thickness: The thickness of the lens at its center.
        fresnel_approximation: If True, approximate the spherical surface by a
            paraboloid, `sqrt(2 R depth)`; otherwise use the sphere,
            `sqrt(R**2 - (R - depth)**2)`.

    Returns:
        The radius of the lens at the depth, 0 outside the lens.
    """
    if not np.abs(depth) < lens_thickness:
        return 0.0
    if fresnel_approximation:
        return float(np.sqrt(2 * radius_of_curvature * depth))
    return float(np.sqrt(radius_of_curvature**2 - (radius_of_curvature - depth) ** 2))


def transparent_plate_phase(
    x_axis: PositionAxis,
    refractive_index: float,
    thickness: float,
    radius: float,
    wavelength: float,
    scale_down: bool,
) -> AlgebraicSignal:
    """Return the phase signal of a thin transparent plate of finite transverse radius.

    The plate delays the field within `radius` of the center of the sampling window by
    the phase `(n - 1) k0 thickness`, relative to the vacuum around it.

    Args:
        x_axis: The position axis.
        refractive_index: The refractive index `n` of the plate.
        thickness: The thickness of the plate.
        radius: The transverse radius of the plate.
        wavelength: The vacuum wavelength.
        scale_down: If True, reduce the phase modulo `2 pi`, which leaves `e^(i f)`
            unchanged but keeps the phases of the signal small.

    Returns:
        The phase signal, on the position axis.
    """
    phase = (refractive_index - 1) * vacuum_wavenumber(wavelength) * thickness
    if scale_down:
        phase = np.mod(phase, 2 * np.pi)
    center = x_axis.sampling_window_length / 2
    return AlgebraicSignal(
        x_axis, lambda x: np.where(np.abs(x - center) <= radius, phase, 0.0)
    )


def free_space_propagator_phase(
    k_axis: AngularWavenumberAxis, distance: float, wavelength: float
) -> QuadraticSignal:
    """Return the paraxial phase `-k**2 distance / (2 k0)` of a free propagation.

    Args:
        k_axis: The angular wavenumber axis of the angular spectrum.
        distance: The propagation distance.
        wavelength: The wavelength in the medium of the propagation.

    Returns:
        The quadratic phase signal, on the angular wavenumber axis.
    """
    return QuadraticSignal(
        k_axis, alpha=-distance / (2 * vacuum_wavenumber(wavelength))
    )
