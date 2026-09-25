"""Exploration of the parameter region in which the sampled phases are resolved.

The phases of a lens slice and of the free propagations over a slice, the lens and the
focal length are sampled on the transverse or angular wavenumber grid. A smooth phase,
like the propagators', aliases in `e^(i f)` once it changes by more than `pi` between
neighboring samples; the phase of a lens slice is a step at the edge of the slice
instead. Their spectra show where the energy sits. Two parameter sets are compared:
the experiment of the paper with its lengths scaled by 1.493 instead of `1e-3` and a beam
FWHM of 20 instead of 25 length units, and one with fewer, thicker slices of a stronger
lens and unscaled phases.
"""

# %%
import numpy as np
from python_wave_optics.elements import free_space_propagator_phase
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.visualization import plot_wavefunction

LENGTH_SCALE = 1.493


def experiment(**overrides) -> ExperimentParameters:
    """Return the parameters of the paper at the length scale, with overrides."""
    parameters = {
        "vacuum_wavelength": 1e-6,
        "beam_FWHM": 20e-3 * LENGTH_SCALE,
        "focal_length": 200e-3 * LENGTH_SCALE,
        "refractive_index": 1.25,
        "propagation_after_lens": 1.5 * 200e-3 * LENGTH_SCALE,
        "transverse_length": 100e-3 * LENGTH_SCALE,
        "num_of_steps_after_lens": 300,
        "lens_slices": 10000,
        "num_qubits": 6,
        "max_delta": 0.01,
        "lens_reverse_order": True,
        "fresnel_approximation": True,
        "scale_down_phases": True,
        "direct_propagator": True,
    }
    return ExperimentParameters(**{**parameters, **overrides})


def max_phase_step(phases: np.ndarray) -> float:
    """Return the largest phase change between neighboring samples, cyclically."""
    return float(np.max(np.abs(np.diff(phases, append=phases[:1]))))


def show_phase(phases: np.ndarray, title: str) -> None:
    """Plot a phase signal, `e^(i f)` and their spectra, and its largest phase step."""
    print(
        f"{title}: largest phase change between neighbors {max_phase_step(phases):.3g}"
    )
    factor = np.exp(1j * phases)
    for values, label in [
        (phases, "phase"),
        (np.fft.fft(phases, norm="ortho"), "spectrum of the phase"),
        (factor, "e^(i phase)"),
        (np.fft.fft(factor, norm="ortho"), "spectrum of e^(i phase)"),
    ]:
        fig, _ = plot_wavefunction(values, normalize=False)
        fig.suptitle(f"{title}: {label}")


def explore(params: ExperimentParameters, lens_slice: int) -> None:
    """Show the phases of a lens slice and of the free propagations of an experiment."""
    show_phase(params.lens_signals[lens_slice].data, f"lens slice {lens_slice}")
    phase_per_slice = (
        (params.refractive_index - 1)
        / params.vacuum_wavelength
        * params.lens_slice_thickness
    )
    print(f"(n - 1) / wavelength * slice thickness = {phase_per_slice}")
    for distance, name in [
        (params.lens_slice_thickness, "slice thickness"),
        (params.lens_thickness, "lens thickness"),
        (params.focal_length, "focal length"),
    ]:
        propagator = free_space_propagator_phase(
            params.k_axis, distance, params.vacuum_wavelength
        )
        show_phase(propagator.data, f"propagation over the {name}")


# %% The parameters of the paper, at the length scale

explore(experiment(), lens_slice=3)

# %% Fewer, thicker slices of a stronger lens, with unscaled phases

explore(
    experiment(
        refractive_index=1.5,
        lens_slices=100,
        max_delta=0.1,
        scale_down_phases=False,
    ),
    lens_slice=0,
)
