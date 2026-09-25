# User Guide

The app consists of the QuTiP backend, [`QutipBackend`][wave_optics_propagation_qutip.backend.QutipBackend], and the scripts in `scripts/`. Everything else, i.e. the parameters of an experiment, the simulation loop, the storage of runs, the classical references and the analysis quantities, is imported from [qiu-classical-simulation](../qiu-classical-simulation/index.md), whose modules are referred to below as `qiu_classical_simulation.wave_optics.<module>`.

## The experiment

The experiment is the one of the Qiskit app, [wave_optics_propagation](../wave_optics_propagation/index.md), whose [User Guide](../wave_optics_propagation/user-guide.md) describes it in depth. In short:

- A Gaussian beam, given by the FWHM of its intensity and centered in a transverse window of `transverse_length`, is sampled on `2**num_qubits` points and enters a plano-convex lens that fills the window, of radius of curvature `R = focal_length (refractive_index - 1)`.
- The lens is sliced along the optical axis into `lens_slices` transparent plates, whose transverse radii follow the spherical surface or, with `fresnel_approximation`, a paraboloid. Each slice delays the field within its radius by the phase `(n - 1) k0 t / N` of the lens thickness `t`, the `N` slices and `k0 = 2 pi / vacuum_wavelength`, reduced modulo `2 pi` with `scale_down_phases`.
- In the forward order, the beam enters through the convex surface and passes the slices from the vertex, `0, ..., N-1`; with `lens_reverse_order`, it enters through the plane side and passes them as `N-1, ..., 0`.
- `qiu_classical_simulation.wave_optics.simulation.simulate` applies the phase of each slice, skipping constant ones, and the free propagation over its thickness, and then propagates freely behind the lens in `num_of_steps_after_lens` equal steps, taking a named snapshot after each slice and step.
- Free propagation multiplies the angular spectrum, the orthonormal DFT of the field, by the paraxial phase `-k**2 dz / (2 k0)`: directly with `direct_propagator`, and otherwise with the sample-based phase protocol.

The phase protocol applies `e^(i f)` for a real signal `f` of one sign, decomposed as `f = alpha |phi|**2` with the sum `alpha` of its samples and the normalized state `phi = sqrt(f / alpha)`, in cycles of equal phases `delta` of magnitude at most `max_delta`. Each cycle, post-selected on its success, maps the amplitudes `psi_j` to `psi_j (1 + (e^(i delta) - 1) |phi_j|**2)`, renormalized, which is `e^(i delta |phi_j|**2) psi_j` up to `O(delta**2)`. Smaller `max_delta` approximates the phases better and makes the success of all cycles more likely.

## The QuTiP backend

`simulate(parameters, backend)` delegates the application of each phase to a `qiu_classical_simulation.wave_optics.simulation.PropagationBackend`, which returns operations on states: functions mapping the amplitudes of a state to the new amplitudes and their probability of success. [`QutipBackend`][wave_optics_propagation_qutip.backend.QutipBackend] implements its two methods with QuTiP operators on kets, of the dimension `d = 2**num_qubits` of the field.

### The sample-based phase protocol

[`sample_based_phase(signal, max_delta)`][wave_optics_propagation_qutip.backend.QutipBackend.sample_based_phase] decomposes the signal with `qiu_classical_simulation.wave_optics.phase_protocol.decompose` into `alpha` and the amplitudes of `|phi>`, and slices `alpha` with `slice_phase` into the equal phases `delta`. Each cycle acts on the ket `|phi> (x) |psi>` of an ancilla register, prepared in `|phi>`, and the field `|psi>`:

1. The partial phase, [`partial_phase_operator(delta, d)`][wave_optics_propagation_qutip.backend.partial_phase_operator], multiplies the basis states `|l> (x) |j>` in which both registers agree, `l == j`, by `e^(i delta)`, and leaves all others unchanged. It is the diagonal operator of dimensions `[[d, d], [d, d]]` with `e^(i delta)` at every `(d + 1)`-th entry.
2. Projecting the ancilla onto `<phi|`, with `tensor(phi.dag(), qeye(d))`, keeps the successful outcome of the cycle: the unnormalized ket of the field.
3. Its squared norm is the probability of success of the cycle; the ket is renormalized and the probability multiplied into the total.

As all phases of one signal are equal, one operator, the projection times the partial phase, serves all its cycles. The projection onto `<phi|` stands for what the circuits of the Qiskit backend do: un-preparing `|phi>` and measuring the ancilla in `|0...0>`.

### The direct propagator

[`direct_momentum_phase(signal)`][wave_optics_propagation_qutip.backend.QutipBackend.direct_momentum_phase] applies the quadratic phase `f` on the angular wavenumber axis, in the FFT ordering, as the operator `F^dagger e^(i f) F`, with the orthonormal DFT `F` of [`dft_operator(d)`][wave_optics_propagation_qutip.backend.dft_operator]: the dense matrix that applies `numpy.fft.fft(..., norm="ortho")`. It is unitary, so its probability of success is 1.

### Compared with the Qiskit backend

Both backends implement the same protocol and the same propagator, and their snapshots and success probabilities agree up to rounding. The QuTiP backend needs no circuits: the ancilla is prepared in `|phi>` directly, each cycle is a single operator applied to a ket of `d**2` entries, and the DFT is a dense matrix. On the few qubits of the transverse field, this is faster than the statevector simulation of the circuits. `qiu_classical_simulation.wave_optics.simulation.ExactBackend` applies the phases exactly and is the reference of both.

## The scripts

The scripts in `scripts/` are section-based (`# %%`): they run as a whole, e.g. with `uv run python`, or cell by cell, e.g. in the interactive window of VS Code. They find the app's directory from their own path, `APP_DIR`, so they run from any working directory.

### `simulate.py`

Simulates one experiment with the QuTiP backend and stores the run, with the same command line as the Qiskit app, `qiu_classical_simulation.wave_optics.cli.parse_parameters`: the experiment of the paper by default, varied by the options

| Option | Default | Sets |
| --- | --- | --- |
| `--max-delta` | required | `max_delta`, the maximum phase per cycle. |
| `--direct-propagator` | off | `direct_propagator`: free propagation directly instead of with the protocol. |
| `--reverse-order` | off | `lens_reverse_order`: the beam enters through the plane side. |
| `--fresnel-approximation` | off | `fresnel_approximation`: a paraboloid instead of the sphere. |
| `--num-qubits` | 7 | `num_qubits`. |
| `--lens-slices` | 10000 | `lens_slices`. |
| `--steps-after-lens` | 300 | `num_of_steps_after_lens`. |
| `--results-dir` | the app's `.result/` | The directory of the run folders. |

It prints the parameters and their validity problems, simulates with a progress bar per loop, stores the run with `save_experiment` in `<results-dir>/<uuid>/`, and prints the folder and the total probability of success. The SLURM jobs in `scripts/cluster/` run the Qiskit app; this script runs the same way, with its path in place of the Qiskit app's.

### `analysis.py`

Analyzes a single run against the thin lens and the classical numerics. `EXPERIMENT_FOLDER` is the run folder, by default one of the runs of December 2025 in `.result/13-lens-simulation-after-classical-numerics/`, and `LEGACY_DEFAULTS` are the parameters those runs did not store, `fresnel_approximation` and `scale_down_phases`, both `True`. The script

- plots the beam waist, twice the standard deviation of the intensity (`qiu_classical_simulation.wave_optics.analysis.beam_waist`), of every lens and free space snapshot against the propagation distance, together with the beam waist behind an ideal thin lens of the same focal length at the principal plane (`thin_lens_reference_states`);
- plots `STEPS_TO_PLOT` (5) snapshots behind the lens, evenly spaced over the free space steps, with `qiu_classical_simulation.wave_optics.visualization.plot_wavefunction`, their magnitude overlaid with the thin lens profile and with the magnitude of the classical numerics at the same distance behind the lens, `classical_numerics_simulation(params, distance)`: the exact split-step field of the same slicing.

The figures are shown, not saved, and need no LaTeX.

## Results

`.result/` in the app's directory, not committed, holds the runs, one folder per run. The current `simulate.py` stores them as `.result/<uuid>/`, each with `initial_parameters.json`, the parameters with the derived lens geometry and the scalar results, and `results.npz`, the snapshots by name. `qiu_classical_simulation.wave_optics.storage` reads them: `load_experiment(folder)` returns the parameters and the result, and `run_folders(results_dir)` the run folders of a directory, sorted by name. `save_experiment` never overwrites a stored run: it raises a `FileExistsError` for a uuid stored already, e.g. of parameters copied with `dataclasses.replace`, which keeps the uuid; such a copy needs a new one, `replace(parameters, ..., uuid=uuid.uuid4().hex)`.

### Runs of December 2025

`.result/13-lens-simulation-after-classical-numerics/` holds the runs of the former script, by their timestamps:

| Run | Order | `max_delta` | Slices |
| --- | --- | --- | --- |
| `2025-12-10_07-53-57` | normal | 0.01 | 10000 |
| `2025-12-10_08-16-27` | reverse, of the previous | 0.01 | 10000 |
| `2025-12-10_14-34-19` | normal | 0.01 | 10000 |
| `2025-12-10_14-47-52` | reverse | 0.01 | 10000 |
| `2025-12-10_15-02-34` | normal | 0.1 | 1000 |
| `2025-12-10_15-04-13` | reverse | 0.1 | 1000 |
| `2025-12-10_15-10-05` | normal | 0.1 | 10000 |
| `2025-12-10_15-12-36` | reverse | 0.1 | 10000 |
| `2025-12-10_15-16-37` | normal | 0.01 | 1000 |
| `2025-12-10_15-33-18` | normal | 0.2 | 10000 |
| `2025-12-10_15-35-45` | reverse | 0.2 | 10000 |

All of them have 6 qubits, a beam of 20 um FWHM and 10 free space steps; `analysis.py` analyzes `2025-12-10_15-12-36` by default. The former script used the Fresnel approximation and phases reduced modulo `2 pi`, which the runs did not store, so they are read with `defaults={"fresnel_approximation": True, "scale_down_phases": True}`, the `LEGACY_DEFAULTS` of `analysis.py`. The storage also reads the other conventions of former versions:

- Snapshots stored as column vectors are flattened.
- The former keys `timestamp` and `reverse_order` are read as `experiment_datetime` and `lens_reverse_order`.
- A missing `direct_propagator` is `True`, which the former code always used, and a missing `uuid` is empty.
- Any other missing parameter raises a `KeyError` naming it.

The runs from `2025-12-10_14-34-19` on store the order as `reverse_order`. The first two runs, `2025-12-10_07-53-57` and `2025-12-10_08-16-27`, do not, so they also need `lens_reverse_order` among the defaults, `False` for the normal and `True` for the reverse order.

## Tests

The tests check that the partial phase acts exactly where the registers agree, that the DFT operator is NumPy's orthonormal FFT, that the backend's phase protocol matches the closed form of the post-selected cycles, state and success probability, that its direct propagator matches the exact one, and that a small simulated experiment approaches the exact one for small `max_delta`. From the repository root:

```sh
uv run pytest apps/wave_optics_propagation_qutip
```
