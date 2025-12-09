# %%

import cupy as cp
import numpy as np
import qutip as qt
import qutip_cuquantum as qtc
from cuquantum.densitymat import WorkStream
from tqdm import tqdm

from wave_optics_propagation_qutip.helpers import apply_phase_protocol

# %%
try:
    num_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"CUDA is available! Number of GPUs: {num_gpus}")
    print(f"Runtime version: {cp.cuda.get_local_runtime_version()}")

    # enable GPU for QuTip
    # stream = WorkStream()
    # qtc.set_as_default(stream)  # Set cuQuantum as the default backend for QuTip

    # qtc.set_as_default()  # Set cuQuantum as the default backend for QuTip
except cp.cuda.runtime.CUDARuntimeError:
    print("CUDA is NOT available.")

# print(qt.settings)
# qt.about()


# %%


# %% Stuff from the previous code

JUPYTER_NAME = "13-lens-simulation-after-classical-numerics"

from datetime import datetime

import matplotlib.pyplot as plt
import scipy.constants as constants
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector, partial_trace

# from qiskit_aer_encore.simulator import generate_aer_simulator
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
    PositionDomainEvolutionQuadratic,
)
from qiskit_hamiltonian_simulation.time_independent.sample_based import (
    KineticEvolutionSampleBased,
    PotentialEvolutionSampleBased,
)
from qiskit_phase_propagator.sample_based_manual import (
    phase_propagate_state_with_arbitrary_signal,
)

# from qiskit_phase_propagator.sample_based import (
#     QuadraticSignalSampleBasedPhasePropagator,
# )
from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import (
    AngularWavenumberAxis,
    MomentumAxis,
    PositionAxis,
)
from qiskit_signals.quantum_signal import GenericQuantumSignal, QuadraticQuantumSignal
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol
from wave_optics_propagation.analytics import (
    free_space_propagated_gaussian_wavefront,
    gaussian_signal,
    propagated_gaussian_wavefront_hitting_lens,
)
from wave_optics_propagation.big_matrix_version import big_matrix_circ
from wave_optics_propagation.elements import (
    free_space_signal_generator,
    radius_of_convex_planar_lens_as_a_func_of_z,
    thin_lens_signal_generator,
    thin_transparent_plate_signal_generator,
)
from wave_optics_propagation.storage import (
    save_initial_parameters,
    save_numpy_results,
)
from wave_optics_propagation.visualization import plot_wavefunction

experiment_datetime = datetime.now()

### Initial parameters

# beam parameters
# vacuum_wavelength = 1e-6
vacuum_wavelength = 1e-6
beam_FWHM = 20e-3 * 1e-3

# lens parameters
focal_length = 200e-3 * 1e-3
# lens_diameter = 25e-3
# refractive_index = 1.5
refractive_index = 1.25

# free space propagation parameters
propagation_after_lens = 1.5 * focal_length

# simulation parameters
transverse_length = 100e-3 * 1e-3  # transverse simulation window
num_of_steps_after_lens = 10
lens_slices = 100
num_qubits = 8
max_delta = 0.01

# lens_diameter = 50e-3 * 1e-3
lens_diameter = transverse_length
### Constants

c = constants.c
# hbar = constants.hbar
### Derived parameters

# geometry
radius_of_curvature = focal_length * (refractive_index - 1)
lens_radius = lens_diameter / 2
lens_thickness = radius_of_curvature - np.sqrt(radius_of_curvature**2 - lens_radius**2)


k_0 = 2 * constants.pi / vacuum_wavelength
reduced_wavelength = vacuum_wavelength / refractive_index

dimension = 2**num_qubits
delta_x = transverse_length / dimension

gaussian_mean = transverse_length / 2
gaussian_beam_waist = beam_FWHM / np.sqrt(2 * np.log(2))
# gaussian_sigma = gaussian_beam_waist / np.sqrt(2)

# vacuum_rayleigh_length = (np.pi * gaussian_beam_waist**2) / vacuum_wavelength

lens_slice_thickness = lens_thickness / lens_slices
step_size_after_lens = propagation_after_lens / num_of_steps_after_lens
print(delta_x, vacuum_wavelength)
### Derived objects

lens_slice_positions = (
    np.linspace(0, lens_thickness, lens_slices, endpoint=False)
    + lens_slice_thickness / 2
)  # choosing the midpoint in each transverse slice

lens_transverse_radii = [
    radius_of_convex_planar_lens_as_a_func_of_z(
        radius_of_curvature, z, lens_thickness, fresnel_approximation=True
    )
    for z in lens_slice_positions
]
print(lens_thickness)
print(lens_slice_positions)
print(lens_transverse_radii)
### Approximations checks

assert (
    gaussian_beam_waist > 10 * vacuum_wavelength
)  # ensure paraxial approximation validity
### Axes

x_axis = PositionAxis(
    num_qubits=num_qubits, delta_x=delta_x, encoding=EncodingType.UNSIGNED
)
k_axis = AngularWavenumberAxis.from_position_axis(x_axis)
# p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)
### Initial beam profile

# def psi_signal_function(x):
#     norm_factor = 1 / (gaussian_sigma * (2 * constants.pi) ** 0.25)
#     return norm_factor * np.exp(-(((x - gaussian_mean) / gaussian_beam_waist) ** 2))


# def psi_signal_function(x):
#     return np.where(
#         (x > 1 / 3) & (x < 2 / 3),
#         1,
#         0,
#     )


# def psi_signal_function(x):
#     signal = np.zeros(dimension)
#     middle = dimension // 2
#     signal[middle - 20 : middle - 15] = 1
#     signal[middle + 15 : middle + 20] = 1
#     return signal

# psi_flat = Statevector.from_label("+" * num_qubits)

# psi = GenericQuantumSignal(x_axis, psi_signal_function)

psi = gaussian_signal(x_axis, gaussian_beam_waist, gaussian_mean)

# print(np.linalg.norm(psi.data) ** 2)
print(gaussian_beam_waist, beam_FWHM, gaussian_mean)
# plot_wavefunction(psi.data, normalize=False)

### Lens potentials

lens_signals = [
    thin_transparent_plate_signal_generator(
        x_axis=x_axis,
        refractive_index=refractive_index,
        thickness=lens_slice_thickness,
        radius=lens_radius,
        wavelength=vacuum_wavelength,
        scale_down=True,
    )
    for lens_radius in lens_transverse_radii
]
NUM_OF_PROFILES_TO_PLOT = 5

# for i in range(NUM_OF_PROFILES_TO_PLOT):
#     plot_wavefunction(lens_signals[i].data, normalize=False)
# plot_wavefunction(lens_signals[-1].data, normalize=False)
sum_signal = np.zeros_like(lens_signals[0].data)
for signal in lens_signals:
    sum_signal += signal.data

# plot_wavefunction(sum_signal, normalize=False)

# %%apply_phase_protocol


# %% New simulation code

current_state = qt.Qobj(psi.data / np.linalg.norm(psi.data))

# plot_wavefunction(current_state.full(), normalize=False)

snapshots = dict([])

snapshots[f"step_{0}"] = current_state.full()

inside_lens_propagation_remained = lens_thickness
total_lenses_simulated = 0
### Lens
for i, lens_radius in enumerate(lens_transverse_radii):
    print(f"Doing lens slice {i + 1}/{lens_slices}")

    lens_signal = lens_signals[i]
    sample_based_lens_signal = (
        ArbitrarySignalForSampleBasedProtocol.from_generic_signal(lens_signal)
    )

    if not np.isclose(np.std(sample_based_lens_signal.data), 0):
        print(f"max value in signal: {np.max(np.abs(sample_based_lens_signal.data))}")
        print(
            f"sum of amplitude square: {np.sum(np.abs(sample_based_lens_signal.data) ** 2)}"
        )

        current_state = apply_phase_protocol(
            current_state, sample_based_lens_signal, max_delta
        )

        total_lenses_simulated += 1

    else:
        print("Skipped the lens slice because it is a flat phase.")

    # propagator_signal = free_space_signal_generator(
    #     k_axis=k_axis,
    #     delta_t=lens_slice_thickness / c,
    #     wavelength=vacuum_wavelength,
    #     c=constants.c,
    # )
    # propagator = MomentumDomainEvolutionQuadratic(quadratic_signal=propagator_signal)

    # current_state = current_state.evolve(propagator)
    # inside_lens_propagation_remained -= lens_slice_thickness

    snapshots[f"step_lens_{i}"] = current_state.full()

# # remaining propagation once the lens slice grows larger than the simulation window
# while inside_lens_propagation_remained > 0:
#     propagator_signal = free_space_signal_generator(
#         k_axis=k_axis,
#         delta_t=inside_lens_propagation_remained / c,
#         wavelength=reduced_wavelength,
#         c=constants.c,
#     )
#     propagator = MomentumDomainEvolutionQuadratic(quadratic_signal=propagator_signal)

#     current_state = current_state.evolve(propagator)
#     inside_lens_propagation_remained -= inside_lens_propagation_remained

# snapshots["after_lens"] = current_state

# for i in range(num_of_steps_after_lens):
#     print(f"Doing free space step {i + 1}/{num_of_steps_after_lens}")

#     propagator_signal = free_space_signal_generator(
#         k_axis=k_axis,
#         delta_t=step_size_after_lens / c,
#         wavelength=vacuum_wavelength,
#         c=constants.c,
#     )
#     propagator = MomentumDomainEvolutionQuadratic(quadratic_signal=propagator_signal)

#     current_state = current_state.evolve(propagator)
#     snapshots[f"step_after_lens_{i + 1}"] = current_state

# snapshots["final"] = current_state

# %%


save_numpy_results(snapshots, experiment_datetime, wrapper_folder=JUPYTER_NAME)
initial_parameters = {
    "timestamp": experiment_datetime.isoformat(),
    "vacuum_wavelength": vacuum_wavelength,
    "beam_FWHM": beam_FWHM,
    "focal_length": focal_length,
    "lens_diameter": lens_diameter,
    "refractive_index": refractive_index,
    "lens_thickness": lens_thickness,
    "propagation_after_lens": propagation_after_lens,
    "transverse_length": transverse_length,
    "num_of_steps_after_lens": num_of_steps_after_lens,
    "lens_slices": lens_slices,
    "num_qubits": num_qubits,
    "max_delta": max_delta,
    "total_lenses_simulated": total_lenses_simulated,
}
save_initial_parameters(
    initial_parameters, experiment_datetime, wrapper_folder=JUPYTER_NAME
)
