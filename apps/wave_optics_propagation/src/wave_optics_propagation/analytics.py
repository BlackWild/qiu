import numpy as np
from qiskit_signals.quantum_axis import GenericAxis
from qiskit_signals.quantum_signal import GenericQuantumSignal

import wave_optics_propagation.BeamProp_Script as bs


def free_space_propagated_gaussian_wavefront(w0, z, wavelength, refractive_index):
    """Calculate the beam waist of a Gaussian beam after propagation in free space.

    Parameters:
    w0 : float
        Initial beam waist (at z=0) in meters.
    z : float
        Propagation distance in meters.
    wavelength : float
        Wavelength of the light in meters.

    Returns:
    float
        Beam waist after propagation in meters.
    """

    mat = bs.prop(z)
    adjusted_wavelength = wavelength / refractive_index

    R_z, w_z = bs.q1_inv_func(0, w0, adjusted_wavelength, mat)

    return R_z, w_z


def propagated_gaussian_wavefront_hitting_lens(
    w0, z, f, wavelength, refractive_index, position_of_beam_waist=0.0
):
    """Calculate the beam waist of a Gaussian beam after hitting a lens and propagating for a distance.

    Parameters:
    w0 : float
        Initial beam waist (at z=0) in meters.
    z : float
        Propagation distance after the lens in meters.
    f : float
        Focal length of the lens in meters.
    wavelength : float
        Wavelength of the light in meters.
    refractive_index : float
        Refractive index of the medium.

    Returns:
    float
        Beam waist after the lens in meters.
    """

    mat1 = bs.lens(f)
    mat2 = bs.prop(z)
    total_mat = bs.mult(mat2, mat1)  # unintuitive but correct order

    adjusted_wavelength = wavelength / refractive_index

    R_f, w_f = bs.q1_inv_func(
        position_of_beam_waist, w0, adjusted_wavelength, total_mat
    )

    return R_f, w_f


def gaussian_wavefront_generator(w0, mean):
    def gaussian_func(x):
        return np.exp(-((x - mean) ** 2) / (w0**2))

    return gaussian_func


def gaussian_signal(x: GenericAxis, w0: float, mean: float) -> GenericQuantumSignal:
    function = gaussian_wavefront_generator(w0, mean)
    signal = GenericQuantumSignal(x, function)
    return signal
