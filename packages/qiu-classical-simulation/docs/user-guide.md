# User Guide

The subpackage [`wave_optics`][qiu_classical_simulation.wave_optics] of `qiu-classical-simulation` simulates one experiment: a Gaussian beam passing a plano-convex lens and propagating freely behind it, in the paraxial approximation and in one transverse dimension. This guide explains the physical model, then each of its modules, in the order a simulation uses them:

| module                                                                 | contents                                                                  |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [`gaussian_beam`][qiu_classical_simulation.wave_optics.gaussian_beam]                    | Gaussian beams through ABCD systems, the Gaussian field on an axis         |
| [`elements`][qiu_classical_simulation.wave_optics.elements]                              | the phase signals of lens slices and of free propagation                  |
| [`parameters`][qiu_classical_simulation.wave_optics.parameters]                          | `ExperimentParameters`, the experiment and everything derived from it     |
| [`simulation`][qiu_classical_simulation.wave_optics.simulation]                          | `simulate`, the backends applying the phases                              |
| [`phase_protocol`][qiu_classical_simulation.wave_optics.phase_protocol]                  | the arithmetic of the sample-based phase protocol                         |
| [`result`][qiu_classical_simulation.wave_optics.result], [`storage`][qiu_classical_simulation.wave_optics.storage] | the snapshots of a run, stored one folder per run                 |
| [`classical_numerics`][qiu_classical_simulation.wave_optics.classical_numerics]          | the classical references: split-step numerics and thin lens analytics     |
| [`analysis`][qiu_classical_simulation.wave_optics.analysis]                              | beam waists, the principal plane, the thin lens references along the way |
| [`visualization`][qiu_classical_simulation.wave_optics.visualization]                    | `plot_wavefunction`                                                       |
| [`cli`][qiu_classical_simulation.wave_optics.cli]                                        | the command line of the simulation scripts                                |

## The physical model

The transverse field `psi(x)` is sampled on `2**num_qubits` points of a window of length `transverse_length`, a position axis of [qiu-signals](../../qiu-signals/) in the `NATURAL` ordering, from `0` to `transverse_length - delta_x`. The beam is centered in the window, and the lens fills it: its diameter is `transverse_length`. The window is periodic, as the discrete Fourier transform makes it, so a field reaching its edges reenters from the other side.

The lens is sliced along the optical axis into `lens_slices` slices of equal thickness `dz`. Each slice is a thin transparent plate of the radius of the lens at the depth of the slice's midpoint. The beam passes a slice in two steps:

1. The plate multiplies the field by `e^(i f(x))`, with the phase `f(x) = (n - 1) k0 dz` within the radius of the plate around the center of the window and `0` outside, `k0 = 2 pi / wavelength` being the vacuum wavenumber. This is the extra optical path of the glass relative to vacuum.
2. The field propagates freely over `dz`: its angular spectrum `psi(k)`, the orthonormal DFT of the field, is multiplied by `e^(i f(k))` with the paraxial (Fresnel) phase `f(k) = -k**2 dz / (2 k0)` on the angular wavenumber axis.

Behind the lens, the field propagates freely over `propagation_after_lens`, in `num_of_steps_after_lens` equal steps. Free propagation always uses the vacuum wavelength, also within the lens, whose effect is entirely in the phases of the plates.

All lengths are in the same unit, which the defaults of the command line take as meters. The phases are in radians.

## Gaussian beams and ABCD matrices

[`gaussian_beam`][qiu_classical_simulation.wave_optics.gaussian_beam] holds the analytic side. A Gaussian beam of waist radius `w0`, a distance `d` behind its waist, has the complex beam parameter `q = d + i z_R`, with the Rayleigh range [`rayleigh_range`][qiu_classical_simulation.wave_optics.gaussian_beam.rayleigh_range]`(w0, wavelength) = pi w0**2 / wavelength`. A system of ray transfer matrix `[[A, B], [C, D]]` maps it to `q' = (A q + B) / (C q + D)`, and the radius of curvature `R` of the wavefront and the beam radius `w` follow from `1 / q' = 1 / R - i wavelength / (pi w**2)`.

- [`propagation_matrix(d)`][qiu_classical_simulation.wave_optics.gaussian_beam.propagation_matrix] and [`thin_lens_matrix(f)`][qiu_classical_simulation.wave_optics.gaussian_beam.thin_lens_matrix] are the matrices of a free propagation and of a thin lens. The matrices of a sequence of elements multiply in reverse order, the last element first: a lens followed by a propagation is `propagation_matrix(d) @ thin_lens_matrix(f)`.
- [`beam_after_system(waist, distance_from_waist, wavelength, matrix)`][qiu_classical_simulation.wave_optics.gaussian_beam.beam_after_system] returns `(R, w)` at the exit of a system; `R` is `inf` for a flat wavefront, e.g. at a waist. Its `wavelength` is the one in the medium.
- [`beam_after_free_space`][qiu_classical_simulation.wave_optics.gaussian_beam.beam_after_free_space] and [`beam_after_thin_lens`][qiu_classical_simulation.wave_optics.gaussian_beam.beam_after_thin_lens] take the vacuum wavelength and the refractive index of the medium, and divide the former by the latter. The beam of `beam_after_free_space` starts at its waist; the one of `beam_after_thin_lens` a `distance_from_waist` before the lens, `0` by default.
- [`gaussian_signal(axis, waist, mean)`][qiu_classical_simulation.wave_optics.gaussian_beam.gaussian_signal] is the field `exp(-(x - mean)**2 / waist**2)` as an algebraic signal. With this convention, `waist` is the radius at which the intensity drops to `1/e**2` of its maximum, the `w` of the formulas above.

```python
import numpy as np
from qiu_classical_simulation.wave_optics.gaussian_beam import (
    beam_after_system,
    beam_after_thin_lens,
    propagation_matrix,
    rayleigh_range,
    thin_lens_matrix,
)

waist, wavelength, focal_length = 20e-6, 1e-6, 1e-3
system = propagation_matrix(0.5e-3) @ thin_lens_matrix(focal_length)
by_matrix = beam_after_system(waist, 0.0, wavelength, system)
by_function = beam_after_thin_lens(waist, 0.5e-3, focal_length, wavelength, 1)
assert np.allclose(by_matrix, by_function)

# at the waist, the wavefront is flat and the radius is the waist
radius_of_curvature, beam_radius = beam_after_system(waist, 0.0, wavelength, np.eye(2))
assert radius_of_curvature == np.inf and np.isclose(beam_radius, waist)
assert np.isclose(rayleigh_range(waist, wavelength), np.pi * waist**2 / wavelength)
```

## Optical elements

[`elements`][qiu_classical_simulation.wave_optics.elements] returns the phase signals of the thin elements, as signals of qiu-signals:

- [`convex_planar_lens_radius(radius_of_curvature, depth, lens_thickness, fresnel_approximation)`][qiu_classical_simulation.wave_optics.elements.convex_planar_lens_radius] is the transverse radius of a plano-convex lens at a depth from its vertex: `sqrt(R**2 - (R - depth)**2)` on the sphere, or `sqrt(2 R depth)` on the paraboloid of the Fresnel approximation, and `0` beyond the thickness of the lens.
- [`transparent_plate_phase(x_axis, refractive_index, thickness, radius, wavelength, scale_down)`][qiu_classical_simulation.wave_optics.elements.transparent_plate_phase] is the phase `(n - 1) k0 thickness` within `radius` of the center of the sampling window, `sampling_window_length / 2`, and `0` outside. With `scale_down`, the phase is reduced modulo `2 pi`, which leaves `e^(i f)` unchanged but keeps the phases small, which the phase protocol profits from.
- [`free_space_propagator_phase(k_axis, distance, wavelength)`][qiu_classical_simulation.wave_optics.elements.free_space_propagator_phase] is the paraxial phase `-k**2 distance / (2 k0)` as a `QuadraticSignal` on the angular wavenumber axis, so that backends can apply it exactly as a quadratic phase of the integer indices.

Both phases have one sign, the plate's non-negative and the propagator's non-positive, as the sample-based phase protocol requires.

!!! note "The center of the window"
    `transparent_plate_phase` centers the plate at `sampling_window_length / 2`, which is the center of the window only for a `NATURAL`-ordered axis, whose values run from `0`. The experiment always uses such an axis; for axes of other orderings, whose values are centered at `0`, the plate would be off-center.

## Experiment parameters

[`ExperimentParameters`][qiu_classical_simulation.wave_optics.parameters.ExperimentParameters] is a frozen dataclass of the given parameters of an experiment:

| parameter                 | meaning                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| `vacuum_wavelength`       | the wavelength of the beam in vacuum                                            |
| `beam_FWHM`               | the full width at half maximum of the intensity of the entering beam            |
| `focal_length`            | the focal length of the lens                                                    |
| `refractive_index`        | the refractive index `n` of the lens                                            |
| `propagation_after_lens`  | the distance of free propagation behind the lens                                |
| `transverse_length`       | the length of the transverse window, and the diameter of the lens               |
| `num_of_steps_after_lens` | the number of equal free propagation steps behind the lens                      |
| `lens_slices`             | the number of slices of the lens                                                |
| `num_qubits`              | the number `n` of qubits of the `2**n` transverse samples                        |
| `max_delta`               | the maximum phase per cycle of the sample-based phase protocol                   |
| `lens_reverse_order`      | if `True`, the beam enters through the plane side of the lens                    |
| `fresnel_approximation`   | if `True`, the lens surface is approximated by a paraboloid                      |
| `scale_down_phases`       | if `True`, the phases of the lens slices are reduced modulo `2 pi`               |
| `direct_propagator`       | if `True`, free propagation is applied directly, else with the phase protocol    |
| `experiment_datetime`     | the time of the experiment, `datetime.now()` by default                          |
| `uuid`                    | the ID of the experiment, a new random hex string by default                     |

Everything else derives from them, as cached properties:

| derived                     | value                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `k0`                        | `2 pi / vacuum_wavelength`                                                                                 |
| `radius_of_curvature`       | `focal_length * (n - 1)`, from the lensmaker's equation of a plano-convex lens                             |
| `lens_radius`               | `transverse_length / 2`; `lens_diameter` is `transverse_length`                                            |
| `lens_thickness`            | `R - sqrt(R**2 - lens_radius**2)`, the sag of the spherical cap, also with the Fresnel approximation        |
| `dimension`, `delta_x`      | `2**num_qubits` and `transverse_length / dimension`                                                        |
| `gaussian_beam_waist`       | `beam_FWHM / sqrt(2 ln 2)`, the `1/e**2` intensity radius                                                  |
| `gaussian_mean`             | `transverse_length / 2`, the center of the window                                                          |
| `lens_slice_thickness`      | `lens_thickness / lens_slices`                                                                             |
| `lens_slice_positions`      | the depths of the midpoints of the slices, from the vertex                                                 |
| `lens_transverse_radii`     | the radii of the slices at these depths, from the vertex                                                   |
| `step_size_after_lens`      | `propagation_after_lens / num_of_steps_after_lens`                                                         |
| `x_axis`                    | the `PositionAxis` of the window, `NATURAL`-ordered                                                        |
| `k_axis`                    | the conjugate `AngularWavenumberAxis`, `FFT`-ordered like the orthonormal DFT                               |
| `initial_beam_profile`      | the Gaussian field of the entering beam, an `AlgebraicSignal`, its waist at the entrance                    |
| `initial_state`             | its normalized amplitudes, `complex128`                                                                    |
| `lens_signals`              | the phase signals of the slices, from the vertex                                                           |
| `ordered_lens_signals`      | the same, in the order the beam passes them: reversed for `lens_reverse_order`                             |

The lens must be at least as curved as the window is wide, `radius_of_curvature >= transverse_length / 2`, or its thickness is not defined. The experiment of the paper has `R = 200 um * 0.25 = 50 um` and a window of `100 um`: a hemisphere.

[`validity_problems()`][qiu_classical_simulation.wave_optics.parameters.ExperimentParameters.validity_problems] lists the violated conditions of the sampling and of the paraxial regime, `delta_x <= vacuum_wavelength` and `gaussian_beam_waist >= 10 vacuum_wavelength`, and [`is_valid()`][qiu_classical_simulation.wave_optics.parameters.ExperimentParameters.is_valid] whether there are none. Nothing enforces them: a simulation of invalid parameters runs, e.g. on a coarse grid for a quick test.

The parameters compare equal field by field, including `uuid` and `experiment_datetime`, so two experiments created separately with the same physics are not equal.

## The simulation loop

[`simulate(parameters, backend, progress=None)`][qiu_classical_simulation.wave_optics.simulation.simulate] runs the experiment:

1. It starts from `initial_state`, and takes the snapshot `step_0`.
2. For each slice `i` of `ordered_lens_signals`, it applies the slice's phase with `backend.sample_based_phase(signal, max_delta)`, then the free propagation over `lens_slice_thickness`, and takes the snapshot `step_lens_{i}`. A slice whose phase is constant over the window, e.g. one covering the whole window, only changes the global phase, so its phase is skipped ([`has_phase`][qiu_classical_simulation.wave_optics.simulation.has_phase]); `total_lenses_simulated` counts the slices applied.
3. It takes the snapshot `after_lens`, equal to the last lens snapshot.
4. For each step `j = 1, ..., num_of_steps_after_lens`, it propagates over `step_size_after_lens` and takes the snapshot `step_after_lens_{j}`.
5. It takes the snapshot `final`, equal to the last step.

The free propagations of [`free_propagation`][qiu_classical_simulation.wave_optics.simulation.free_propagation] apply `free_space_propagator_phase` to the angular spectrum: with `backend.direct_momentum_phase(signal)` if `direct_propagator` is set, and otherwise with `backend.sample_based_phase(signal, max_delta)` on the orthonormal DFT of the state ([`to_angular_spectrum`][qiu_classical_simulation.wave_optics.simulation.to_angular_spectrum]), transformed back afterwards ([`from_angular_spectrum`][qiu_classical_simulation.wave_optics.simulation.from_angular_spectrum]).

`progress` optionally wraps the two loops, called with the iterable and a description, `"Lens slices"` and `"Free space steps"`, e.g. `progress=lambda loop, name: tqdm(loop, desc=name)` for progress bars.

## Backends

A [`PropagationBackend`][qiu_classical_simulation.wave_optics.simulation.PropagationBackend] decides how the phases are applied. It implements two methods, each returning a [`PhaseOperation`][qiu_classical_simulation.wave_optics.simulation.PhaseOperation], a function from a [`State`][qiu_classical_simulation.wave_optics.simulation.State], the normalized amplitudes, to the new state and its probability of success:

- `sample_based_phase(signal, max_delta)` applies `e^(i signal)` with the sample-based phase protocol, post-selected on the success of all its cycles, for a real signal of one sign on the axis of the state.
- `direct_momentum_phase(signal)` applies `e^(i signal)` to the angular spectrum of the state, for a `QuadraticSignal` on the `FFT`-ordered angular wavenumber axis; its probability of success is 1.

The operations are created once per phase signal and may be applied many times, e.g. the propagation over the slice thickness to every slice, so a backend should do its expensive preparation, e.g. building circuits or operators, when creating the operation.

[`ExactBackend`][qiu_classical_simulation.wave_optics.simulation.ExactBackend] applies both exactly, by multiplying with `e^(i f)` in the position or the angular spectrum domain, and ignores `max_delta`; every probability of success is 1. It is the classical reference of the quantum backends of the apps, which implement the same interface with Qiskit circuits and QuTiP operators.

## The phase protocol

[`phase_protocol`][qiu_classical_simulation.wave_optics.phase_protocol] holds the arithmetic of the sample-based phase protocol, independent of its implementation, for backends and for the analysis of its cost:

- [`decompose(signal)`][qiu_classical_simulation.wave_optics.phase_protocol.decompose] splits a real signal of one sign as `f = alpha |phi|**2`, with the sum `alpha` of its samples, negative for a non-positive signal, and the normalized amplitudes `phi = sqrt(f / alpha)`. Signals of mixed signs, with a non-zero imaginary part, or vanishing, raise a `ValueError`.
- [`slice_phase(alpha, max_delta)`][qiu_classical_simulation.wave_optics.phase_protocol.slice_phase] slices `alpha` into the fewest equal phases `delta` of magnitude at most `max_delta`, `ceil(|alpha| / max_delta)` of them.
- [`ideal_cycles(psi, phi, deltas)`][qiu_classical_simulation.wave_optics.phase_protocol.ideal_cycles] applies the cycles in closed form: each successful cycle maps `psi_j` to `psi_j (1 + (e^(i delta) - 1) |phi_j|**2)`, renormalized, which is `e^(i delta |phi_j|**2) psi_j` up to `O(delta**2)`. It returns the final state and the product of the probabilities of success of the cycles.

The number of cycles grows with the sum of the samples of the phase, not with its maximum: a plate phase of `0.3` on 100 samples has `alpha = 30`, i.e. 3000 cycles at `max_delta = 0.01`. A smaller `max_delta` makes the protocol more accurate and, since a cycle fails with a probability of order `delta**2`, more likely to succeed overall, but needs proportionally more cycles.

## Results

[`ExperimentResult`][qiu_classical_simulation.wave_optics.result.ExperimentResult] holds the output of `simulate`:

- `snapshots`: the normalized amplitudes of the named snapshots, flattened, in the order taken. The names are those of the loop above; [`lens_snapshot_name`][qiu_classical_simulation.wave_optics.result.lens_snapshot_name] and [`free_space_snapshot_name`][qiu_classical_simulation.wave_optics.result.free_space_snapshot_name] build them, the slices counted from 0 and the steps from 1.
- `total_lenses_simulated`: the number of slices whose phase was applied.
- `total_probability_of_success` and `success_probabilities`: the probability that all cycles succeeded, and the cumulative probability at each snapshot, in the order of `snapshots`. [`success_probability(name)`][qiu_classical_simulation.wave_optics.result.ExperimentResult.success_probability] looks one up by the name of its snapshot; results stored by old versions may have none, and it raises a `ValueError`.
- [`lens_states(lens_slices)`][qiu_classical_simulation.wave_optics.result.ExperimentResult.lens_states] and [`free_space_states(steps)`][qiu_classical_simulation.wave_optics.result.ExperimentResult.free_space_states] return the snapshots after each slice and after each step, which [`propagation_distances`][qiu_classical_simulation.wave_optics.analysis.propagation_distances] pairs with their distances.

## Classical references

[`classical_numerics`][qiu_classical_simulation.wave_optics.classical_numerics] computes what the simulations are compared with:

- [`classical_numerics_simulation(parameters, propagation_after_lens)`][qiu_classical_simulation.wave_optics.classical_numerics.classical_numerics_simulation] is the exact split-step field at a distance behind the lens: the same slices as the simulation, applied exactly, and a single free propagation behind the lens. It equals the snapshots of the `ExactBackend`, with either `direct_propagator`, up to rounding.
- [`propagate_exactly(state, parameters, distance)`][qiu_classical_simulation.wave_optics.classical_numerics.propagate_exactly] is an exact free propagation in vacuum.
- [`thin_lens_simulation(parameters, propagation_before_lens, propagation_after_lens)`][qiu_classical_simulation.wave_optics.classical_numerics.thin_lens_simulation] is the analytic profile behind an ideal thin lens of the same focal length, placed `propagation_before_lens` behind the waist of the beam. It returns the normalized magnitude only, a real Gaussian, not the phase: compare it with `abs` of a field, or through beam waists.

The split-step field includes everything the thin lens neglects: the thickness of the lens, the aberrations of the spherical surface and the diffraction within the lens.

## Storage

[`storage`][qiu_classical_simulation.wave_optics.storage] stores a run in its own folder `<results_dir>/<uuid>/`:

- `initial_parameters.json`: the parameters of [`to_dict()`][qiu_classical_simulation.wave_optics.parameters.ExperimentParameters.to_dict], with the derived `lens_diameter` and `lens_thickness` for reference, together with the scalar results of `ExperimentResult.summary()`;
- `results.npz`: the snapshots, by name.

[`save_experiment(results_dir, parameters, result)`][qiu_classical_simulation.wave_optics.storage.save_experiment] writes a run and returns its folder, [`load_experiment(folder, defaults=None)`][qiu_classical_simulation.wave_optics.storage.load_experiment] reads it back as `(parameters, result)`, and [`load_parameters`][qiu_classical_simulation.wave_optics.storage.load_parameters] and [`load_result`][qiu_classical_simulation.wave_optics.storage.load_result] read either part. [`run_folders(results_dir)`][qiu_classical_simulation.wave_optics.storage.run_folders] lists the folders holding a run, sorted by name.

The storage reads the results of all former versions of the apps, whose folders were named by timestamps:

- Snapshots stored as column vectors are flattened.
- The former keys `timestamp` and `reverse_order` are read as `experiment_datetime` and `lens_reverse_order` ([`LEGACY_KEYS`][qiu_classical_simulation.wave_optics.parameters.LEGACY_KEYS]).
- A missing `direct_propagator` is `True`, which the former code always used, and a missing `uuid` is empty ([`LEGACY_DEFAULTS`][qiu_classical_simulation.wave_optics.parameters.LEGACY_DEFAULTS]).
- Other missing parameters, e.g. `fresnel_approximation` of the runs of December 2025, raise a `KeyError` naming them, and must be given explicitly: `load_experiment(folder, defaults={...})`. Stored values take precedence over the defaults, and unknown keys are ignored.

## Analysis

[`analysis`][qiu_classical_simulation.wave_optics.analysis] computes the quantities of the figures of the paper:

- [`beam_waist(state, x_values)`][qiu_classical_simulation.wave_optics.analysis.beam_waist] is twice the standard deviation of the intensity of a field, which is the waist `w` of a Gaussian beam `exp(-x**2 / w**2)`. It is defined for any field, but agrees with the `w` of the Gaussian formulas only for Gaussian profiles; side lobes, e.g. of aberrations, widen it.
- [`principal_plane_position(parameters)`][qiu_classical_simulation.wave_optics.analysis.principal_plane_position] is the depth, from the entrance, of the principal plane of the lens on the side of the exit, where an equivalent thin lens sits: at `t - t / n` for the forward order, and at the convex vertex, the exit, `t`, for the reverse order.
- [`propagation_distances(parameters)`][qiu_classical_simulation.wave_optics.analysis.propagation_distances] are the distances from the entrance of the snapshots after each slice and each step, in the order of `lens_states` followed by `free_space_states`.
- [`thin_lens_reference_states(parameters)`][qiu_classical_simulation.wave_optics.analysis.thin_lens_reference_states] are the thin lens profiles at these distances: the freely propagating beam before the principal plane, and the beam behind a thin lens at the principal plane after it.
- [`lens_surface(parameters)`][qiu_classical_simulation.wave_optics.analysis.lens_surface] is the depth of the convex surface at each transverse position, measured from the entrance, to draw the lens into an intensity map.

## Visualization and command line

[`plot_wavefunction(psi, plot_size_scale=1, normalize=True, ylim=None)`][qiu_classical_simulation.wave_optics.visualization.plot_wavefunction] plots the magnitude and the phase of a field over its sample indices, side by side, and returns the figure and its two axes, without showing or saving it.

[`cli`][qiu_classical_simulation.wave_optics.cli] is the command line the simulation scripts of the apps share. [`parse_parameters(description, default_results_dir, argv=None)`][qiu_classical_simulation.wave_optics.cli.parse_parameters] parses it into the parameters and the results directory. The physics is fixed to the experiment of the paper: a beam of `25 um` FWHM at `1 um` wavelength, a lens of `200 um` focal length and refractive index `1.25` in a window of `100 um`, followed by `300 um` of free space, in meters, with `scale_down_phases` set. The command line varies the rest:

```text
--max-delta FLOAT          required, the maximum phase per cycle
--direct-propagator        apply free propagation directly (default: with the phase protocol)
--reverse-order            enter the lens through its plane side
--fresnel-approximation    approximate the lens surface by a paraboloid
--num-qubits INT           default 7
--lens-slices INT          default 10000
--steps-after-lens INT     default 300
--results-dir PATH         default: the given default_results_dir
```

## Pitfalls

- **Copies of parameters share their ID.** `dataclasses.replace(parameters, max_delta=0.01)` keeps the `uuid` and `experiment_datetime` of the original, and `save_experiment` refuses to store a second run of the same `uuid` with a `FileExistsError`, rather than overwriting the first. Give the copy a new ID, `dataclasses.replace(parameters, max_delta=0.01, uuid=uuid.uuid4().hex)`.
- **The window is periodic.** A beam that spreads to the edges of the window reenters from the other side, and its `beam_waist` is then meaningless. Keep `propagation_after_lens` and the beam well within the window.
- **The thin lens reference is not the thick lens.** The sliced lens has a thickness and the aberrations of its spherical surface, which the thin lens neglects. For the hemisphere of the paper, the simulated focus lies before the thin lens focus and is several times wider; for thin, weakly curved lenses the two agree closely (see the [Examples](examples.md)).
- **Invalid parameters are not rejected.** Check `validity_problems()` before long runs.
- **The `ExactBackend` ignores `max_delta`.** Its results do not depend on it; use a backend of the protocol, or `ideal_cycles`, to study its effect.
