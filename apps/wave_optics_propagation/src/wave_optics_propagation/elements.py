import numpy as np
import scipy.constants as constants
from qiskit_signals.quantum_axis import (
    AngularWavenumberAxis,
    PositionAxis,
)
from qiskit_signals.quantum_signal import (
    GenericQuantumSignal,
    QuadraticQuantumSignal,
)


def radius_of_convex_planar_lens_as_a_func_of_z(
    radius_of_curvature: float,
    z: float,
    lens_thickness: float,
    fresnel_approximation: bool,
) -> float:
    """Calculates the radius of a convex planar lens as a function of z.

    Args:
        radius_of_curvature: The radius of curvature of the lens.
        z: The distance from the lens center along the optical axis.
        lens_thickness: The thickness of the lens.

    Returns:
        The radius of the lens at distance z. If z is outside the lens thickness, returns 0.
    """
    if fresnel_approximation:
        x = np.sqrt(2 * radius_of_curvature * z)
    else:
        x = np.sqrt(radius_of_curvature**2 - (radius_of_curvature - z) ** 2)

    return x if np.abs(z) < lens_thickness else 0


def thin_transparent_plate_signal_generator(
    x_axis: PositionAxis,
    refractive_index: float,
    thickness: float,
    radius: float,
    wavelength: float,
    scale_down: bool,
) -> GenericQuantumSignal:
    """Generates a quantum signal for a finite transparent plate.

    Args:
        x_axis: The position axis.
        refractive_index: The refractive index of the plate.
        thickness: The thickness of the plate.
        radius: The radius of the plate.
        wavelength: The wavelength of the light.

    Returns:
        A GenericQuantumSignal representing the phase shift introduced by the plate.
    """
    k_0 = 2 * np.pi / wavelength
    phase_shift = (refractive_index - 1) * k_0 * thickness

    if scale_down:

        def signal_function(x):
            return np.where(
                np.abs(x - x_axis.sampling_window_length / 2)
                <= radius,  # TODO: probably should change this to actually compare the distance from the optical axis which depends on the encoding we use among other things
                np.mod(phase_shift, 2 * np.pi),
                0,
            )
    else:

        def signal_function(x):
            return np.where(
                np.abs(x - x_axis.sampling_window_length / 2)
                <= radius,  # TODO: probably should change this to actually compare the distance from the optical axis which depends on the encoding we use among other things
                phase_shift,
                0,
            )

    signal = GenericQuantumSignal(axis=x_axis, signal_function=signal_function)
    return signal


def thin_lens_signal_generator(
    x_axis: PositionAxis, focal_length: float, wavelength: float
) -> QuadraticQuantumSignal:
    """Generates a quantum signal for a thin lens.

    Args:
        x_axis: The position axis.
        focal_length: The focal length of the lens.
        wavelength: The wavelength of the light.

    Returns:
        A QuadraticQuantumSignal representing the phase shift introduced by the lens.
    """
    k_0 = 2 * np.pi / wavelength
    coef_for_lens = k_0 / (2 * -focal_length)
    quadratic_signal = QuadraticQuantumSignal(axis=x_axis, alpha=coef_for_lens)
    return quadratic_signal


def free_space_signal_generator(
    k_axis: AngularWavenumberAxis,
    delta_t: float,
    wavelength: float,
    c: float = constants.c,
) -> QuadraticQuantumSignal:
    """Generates a quantum signal for free space propagation.

    Args:
        k_axis: The angular wavenumber axis.
        delta_t: The time interval for propagation.
        wavelength: The wavelength of the light.
        c: The speed of light in the medium. Default is the speed of light in vacuum.

    Returns:
        A QuadraticQuantumSignal representing free space propagation.
    """
    k_0 = 2 * np.pi / wavelength
    coef = -1 / 2 / k_0 * c * delta_t
    quadratic_signal = QuadraticQuantumSignal(axis=k_axis, alpha=coef)
    return quadratic_signal
