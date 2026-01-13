import numpy as np
import scipy.constants as constants
from wave_optics_propagation.analytics import (
    free_space_propagated_gaussian_wavefront,
    gaussian_signal,
    propagated_gaussian_wavefront_hitting_lens,
)
from wave_optics_propagation.elements import free_space_signal_generator

from wave_optics_propagation_enhanced.parameters import ExperimentParameters


def classical_numerics_simulation(
    params: ExperimentParameters, propagation_after_lens: float
) -> np.ndarray:
    inside_lens_propagation_remained = params.lens_thickness
    signal_data = params.initial_beam_profile.normalized_data.copy()

    for j in range(params.lens_slices):
        # print(f"Doing lens slice {i + 1}/{lens_slices}")

        lens_signal_loop = (
            params.lens_signals[-j]
            if params.lens_reverse_order
            else params.lens_signals[j]
        )

        if not np.isclose(np.std(lens_signal_loop.data), 0):
            # print(f"max value in signal: {np.max(np.abs(lens_signal_loop.data))}")
            # print(
            #     f"sum of amplitude square: {np.sum(np.abs(lens_signal_loop.data) ** 2)}"
            # )
            # continue
            signal_data = signal_data * np.exp(1j * lens_signal_loop.data)
            # total_lenses_simulated += 1

        else:
            # print("Skipped the lens slice because it is a flat phase.")
            pass

        propagator_signal = free_space_signal_generator(
            k_axis=params.k_axis,
            delta_t=params.lens_slice_thickness / constants.c,
            wavelength=params.vacuum_wavelength,
            c=constants.c,
        )

        fourier_transformed = np.fft.fft(signal_data, norm="ortho")
        propagated = fourier_transformed * np.exp(1j * propagator_signal.data)
        signal_data = np.fft.ifft(propagated, norm="ortho")

        inside_lens_propagation_remained -= params.lens_slice_thickness

    # remaining propagation once the lens slice grows larger than the simulation window
    while inside_lens_propagation_remained > 0:
        propagator_signal = free_space_signal_generator(
            k_axis=params.k_axis,
            delta_t=inside_lens_propagation_remained / constants.c,
            wavelength=params.reduced_wavelength,
            c=constants.c,
        )

        fourier_transformed = np.fft.fft(signal_data, norm="ortho")
        propagated = fourier_transformed * np.exp(1j * propagator_signal.data)
        signal_data = np.fft.ifft(propagated, norm="ortho")

        inside_lens_propagation_remained -= inside_lens_propagation_remained

    # for i in range(num_of_steps_after_lens):
    # print(f"Doing free space step {i + 1}/{num_of_steps_after_lens}")

    propagator_signal = free_space_signal_generator(
        k_axis=params.k_axis,
        delta_t=propagation_after_lens / constants.c,
        wavelength=params.vacuum_wavelength,
        c=constants.c,
    )

    fourier_transformed = np.fft.fft(signal_data, norm="ortho")
    propagated = fourier_transformed * np.exp(1j * propagator_signal.data)
    signal_data = np.fft.ifft(propagated, norm="ortho")

    return signal_data


def thin_lens_simulation(
    params: ExperimentParameters,
    propagation_before_lens: float,
    propagation_after_lens: float,
) -> np.ndarray:
    R_z_1, w_z_1 = free_space_propagated_gaussian_wavefront(
        w0=params.gaussian_beam_waist,
        z=propagation_before_lens,
        wavelength=params.vacuum_wavelength,
        refractive_index=1,
    )

    R_z_2, w_z_2 = propagated_gaussian_wavefront_hitting_lens(
        w0=params.gaussian_beam_waist,
        z=propagation_after_lens,
        f=params.focal_length,
        wavelength=params.vacuum_wavelength,
        refractive_index=1,
        position_of_beam_waist=propagation_before_lens,
    )

    signal = gaussian_signal(params.x_axis, w_z_2, params.gaussian_mean)
    return signal.normalized_data
