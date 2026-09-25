# User Guide

The app consists of the Qiskit backend, [`QiskitBackend`][wave_optics_propagation.backend.QiskitBackend], and the scripts in `scripts/`. Everything else, i.e. the parameters of an experiment, the simulation loop, the storage of runs, the classical references and the analysis quantities, is imported from [qiu-classical-simulation](../qiu-classical-simulation/index.md), whose modules are referred to below as `qiu_classical_simulation.wave_optics.<module>`.

## The experiment

A Gaussian beam of wavelength `vacuum_wavelength` passes a plano-convex lens and then propagates freely behind it. The field is one-dimensional in the transverse direction: it is sampled on `2**num_qubits` points of a window of `transverse_length`, the position axis `x_axis` from 0 in steps of `delta_x`, and is a normalized state of `num_qubits` qubits. An experiment is given by `qiu_classical_simulation.wave_optics.parameters.ExperimentParameters`, from which everything else derives:

- The beam enters with its waist at the lens, centered in the window, as the field `exp(-(x - L/2)**2 / w0**2)` of the waist radius `w0 = beam_FWHM / sqrt(2 ln 2)`, i.e. `beam_FWHM` is the full width at half maximum of its intensity (`gaussian_beam_waist`, `initial_state`).
- The lens fills the window, `lens_diameter = transverse_length`. The radius of curvature of its convex surface follows from the lensmaker's equation of a plano-convex lens, `R = focal_length (refractive_index - 1)`, and its thickness at the center is `t = R - sqrt(R**2 - (L/2)**2)`.
- The lens is sliced along the optical axis into `lens_slices` transparent plates of thickness `t / N`. The slice at the depth `d` of its midpoint, measured from the vertex of the convex surface, has the transverse radius `sqrt(R**2 - (R - d)**2)` of the spherical surface, or `sqrt(2 R d)` of a paraboloid with `fresnel_approximation`. Within that radius it delays the field by the phase `(n - 1) k0 t / N` relative to the vacuum around it, `k0 = 2 pi / vacuum_wavelength`; this phase signal on the position axis is its `lens_signals` entry.
- With `scale_down_phases`, the phase of each slice is reduced modulo `2 pi`. This leaves `e^(i f)` unchanged, but keeps the phases, and thus the number of cycles of the phase protocol, small.
- The beam passes the slices in the order of `ordered_lens_signals`. In the forward order, it enters through the convex surface and passes the slices from the vertex, `0, ..., N-1`. With `lens_reverse_order`, it enters through the plane side and passes them as `N-1, ..., 0`.

The simulation, `qiu_classical_simulation.wave_optics.simulation.simulate`, passes each slice, i.e. applies its phase and propagates freely over the slice thickness, and then propagates freely over `propagation_after_lens` in `num_of_steps_after_lens` equal steps. The phase of a slice that is constant over the window, e.g. of one whose radius covers no sample or all of them, is only a global phase and is skipped, while the propagation over the slice thickness is not; `total_lenses_simulated` counts the slices whose phase is applied.

Free propagation over a distance `dz` multiplies the angular spectrum of the field, its orthonormal DFT, by `e^(i f(k))` with the paraxial (Fresnel) phase `f(k) = -k**2 dz / (2 k0)` on the angular wavenumber axis `k_axis`, in the FFT ordering. Within the lens, the field also propagates as in vacuum; the delay of the glass is the phase of the slices.

### Direct and sample-based propagator

`direct_propagator` selects how free propagation is applied:

- The direct propagator applies the quadratic phase to the angular spectrum as a unitary, exactly and always successfully.
- The sample-based propagator applies it with the sample-based phase protocol, like the phases of the lens slices: the field is transformed to its angular spectrum, the protocol applies the phase, which is of one sign, and the result is transformed back.

### The sample-based phase protocol

The protocol applies `e^(i f)` for a real signal `f` of one sign. The signal is decomposed as `f = alpha |phi|**2`, with the sum `alpha` of its samples and the normalized state `phi = sqrt(f / alpha)`, and `alpha` is sliced into the fewest equal phases `delta` of magnitude at most `max_delta`. Each cycle of a phase `delta`, post-selected on its success, maps the amplitudes `psi_j` to `psi_j (1 + (e^(i delta) - 1) |phi_j|**2)`, renormalized, which is `e^(i delta |phi_j|**2) psi_j` up to `O(delta**2)`. See `qiu_classical_simulation.wave_optics.phase_protocol` for its arithmetic and [qiu-quantum-computing](../qiu-quantum-computing/index.md) for its circuits.

Smaller `max_delta` thus approximates the phases better, with more cycles, and each cycle succeeds with a probability closer to 1. The simulations keep only the successful outcomes and multiply up the probabilities of success; the probability that all cycles of an experiment succeed drops with `max_delta` roughly exponentially, as `exp(a max_delta)` with `a < 0`, and the fidelity to the exact field roughly as `1 + a max_delta**2`, the models the batch analysis fits.

### Validity

`ExperimentParameters.validity_problems()` lists the violated conditions of the sampling and of the paraxial regime: a spacing `delta_x` above the wavelength, and a beam waist below 10 wavelengths. The simulation scripts print them as warnings but run anyway.

### The experiment of the paper

The command line of the scripts, `qiu_classical_simulation.wave_optics.cli`, fixes the physical parameters to the experiment of the paper and lets the options vary the rest:

| Parameter | Value |
| --- | --- |
| `vacuum_wavelength` | 1 um |
| `beam_FWHM` | 25 um, a waist radius of about 21.2 um |
| `focal_length` | 200 um |
| `refractive_index` | 1.25, i.e. `R` = 50 um |
| `transverse_length` | 100 um, i.e. a lens of 50 um thickness, a hemisphere |
| `propagation_after_lens` | 300 um |
| `scale_down_phases` | `True` |

With the default 7 qubits, `delta_x` is about 0.78 um; with fewer qubits, it exceeds the wavelength and the scripts warn about it.

### Snapshots and success probabilities

`simulate` returns a `qiu_classical_simulation.wave_optics.result.ExperimentResult` with a snapshot of the normalized field after each operation, named in the order taken:

| Snapshot | Taken |
| --- | --- |
| `step_0` | The entering beam. |
| `step_lens_{i}` | After the `i`-th slice passed, `i = 0, ..., N-1`, in the order passed. |
| `after_lens` | Behind the lens, equal to the last lens snapshot. |
| `step_after_lens_{j}` | After the free propagation step `j = 1, ..., num_of_steps_after_lens`. |
| `final` | The end, equal to the last step. |

`success_probabilities` holds the cumulative probability of success at each snapshot, `success_probability(name)` looks it up by name, and `total_probability_of_success` is the one of the whole experiment. `lens_states(lens_slices)` and `free_space_states(num_of_steps_after_lens)` return the snapshots of the slices and of the steps.

## The Qiskit backend

`simulate(parameters, backend)` delegates the application of each phase to a `qiu_classical_simulation.wave_optics.simulation.PropagationBackend`, which returns operations on states: functions mapping a state to the new state and its probability of success. A backend implements two methods:

- `sample_based_phase(signal, max_delta)`: the post-selected phase protocol applying `e^(i signal)` to a state, for the lens slices and, without `direct_propagator`, the free propagation on the angular spectrum.
- `direct_momentum_phase(signal)`: the exact application of a quadratic phase to the angular spectrum of a state, for the direct propagator, with probability of success 1.

[`QiskitBackend`][wave_optics_propagation.backend.QiskitBackend] implements them with Qiskit:

- [`sample_based_phase`][wave_optics_propagation.backend.QiskitBackend.sample_based_phase] decomposes the signal with `qiu_quantum_computing.phase_propagator.sample_based.sample_based_decomposition`, slices `alpha` evenly and prepares `|phi>` as a `qiu_quantum_computing.preparable_state.PreparableState`. Each cycle, `qiu_quantum_computing.phase_propagator.sample_based_manual.phase_propagation_cycle`, is a statevector simulation of the circuit of the protocol on `2n` qubits: it prepares `|phi>` in a second register, applies the partial phase `e^(i delta)` where both registers agree (by its diagonal), un-prepares `|phi>`, and keeps the renormalized part in which the second register is `|0...0>`, i.e. the successful outcome, together with its probability.
- [`direct_momentum_phase`][wave_optics_propagation.backend.QiskitBackend.direct_momentum_phase] evolves the state with the circuit `MomentumDomainEvolutionQuadratic` of [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/index.md): a change to the momentum basis, Qiskit's inverse QFT, the quadratic phase, and the QFT back.

Its two options set how these circuits are represented, as a `qiu_qiskit_encore.synthesis_method.SynthesisMethod`:

| Option | Default | Represents |
| --- | --- | --- |
| [`state_preparation_method`][wave_optics_propagation.backend.QiskitBackend.state_preparation_method] | `DENSE` | The preparation of `|phi>`. The dense unitary is exact and fast on the few qubits of the transverse field; `DECOMPOSED` simulates the decomposed circuit. |
| [`fourier_method`][wave_optics_propagation.backend.QiskitBackend.fourier_method] | `GATE` | The Fourier transforms of the direct propagator. |

The backend is passed to the simulation like any other; `progress` optionally wraps the loops over the slices and the steps, e.g. with a progress bar:

```python
from pathlib import Path

from qiu_classical_simulation.wave_optics.cli import parse_parameters
from qiu_classical_simulation.wave_optics.simulation import simulate
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from wave_optics_propagation.backend import QiskitBackend

parameters, _ = parse_parameters(
    "",
    Path(".result"),
    ["--max-delta=0.1", "--num-qubits=4", "--lens-slices=4", "--steps-after-lens=3"],
)
loops = []


def progress(loop, name):
    """Record the name of the loop and iterate over it unchanged."""
    loops.append(name)
    return loop


backend = QiskitBackend(state_preparation_method=SynthesisMethod.DECOMPOSED)
result = simulate(parameters, backend, progress=progress)
assert loops == ["Lens slices", "Free space steps"]
assert len(result.snapshots) == 4 + 3 + 3  # with step_0, after_lens and final
```

`qiu_classical_simulation.wave_optics.simulation.ExactBackend` applies the same phases exactly and is the reference of the backend; the QuTiP backend of [wave_optics_propagation_qutip](../wave_optics_propagation_qutip/index.md) applies them as operators on kets, with the same results up to rounding.

## The scripts

The scripts in `scripts/` are section-based (`# %%`): they run as a whole, e.g. with `uv run python`, or cell by cell, e.g. in the interactive window of VS Code. They find the app's directories from their own path, `APP_DIR`, so they run from any working directory. The analyses typeset with LaTeX (`text.usetex`) and save their figures as PDF to the app's `.output/`, which they create.

### `simulate.py`

Simulates one experiment with the Qiskit backend and stores the run. It parses the command line of `qiu_classical_simulation.wave_optics.cli.parse_parameters`, prints the parameters and their validity problems, simulates with a progress bar per loop, stores the run with `save_experiment` in `<results-dir>/<uuid>/`, and prints the folder and the total probability of success.

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

### `forward_analysis.py`

The propagation figure of the paper of a single run. `EXPERIMENT_ID` names the run folder in `.result/`, `RUN_FOLDER`; the comments above it list the local runs by their timestamps and the cluster runs, which have to be copied into `.result/` first. The script

- computes the beam waist, twice the standard deviation of the intensity (`qiu_classical_simulation.wave_optics.analysis.beam_waist`), of every lens and free space snapshot, and of the thin lens references at the same propagation distances (`thin_lens_reference_states`): the analytic profile of the beam behind an ideal thin lens of the same focal length at the principal plane of the lens (`principal_plane_position`), at the depth `t` for the reverse order, i.e. at the vertex of the convex surface where the beam exits, and at `t - t / n` for the forward one;
- plots both beam waists against the propagation distance;
- draws the intensity map through the lens and behind it, with the beam waist behind the lens, the lens surface and sides, the focal point, taken as the distance of the smallest simulated beam waist, and the principal plane;
- saves the map as `wave_propagation.pdf` into the run folder and as `wave_propagation-<order>-<model>.pdf` into `.output/`, with `r` or `nr` for the reverse or forward order and `f` or `nf` with or without the Fresnel approximation;
- prints the focal point and the radius of curvature.

### `batch_analysis.py`

The figures of the paper of the batch over `max_delta`, the runs of `scripts/cluster/batch-run.slurm` in `BATCH_DIR`, `.result/batch/`:

1. It loads the parameters of all runs, drops repetitions of the same parameters (all but `experiment_datetime`, `uuid` and the folder) and sorts them by `max_delta`.
2. For each run, it compares the snapshot `step_after_lens_{s}` of the reference step `s = num_of_steps_after_lens * 2 // 3 + 1`, i.e. two thirds of the way behind the lens, with the classical numerics at the same distance, `classical_numerics_simulation(params, step_size_after_lens * s)`: the exact split-step field of the same slicing. It records their overlap and the success probability of the protocol up to that snapshot.
3. For the runs of the sample-based propagator, without the `NUMS_TO_CUT` (9) of the largest `max_delta`, it fits the fidelity by `1 + a max_delta**2` and the success probability by `exp(a max_delta)`, both through the first `NUMS_TO_KEEP_FOR_FITTING` (8) runs.
4. It saves `overlap_with_classical_numerics.pdf` and `success_probability_vs_max_delta.pdf`, with a logarithmic probability axis, into `.output/`, and prints the coefficients, the largest success probability and the slopes of the success probability in log scale and, through the first five runs, in log-log scale.

### `parameter_validity.py`

An exploration of how well the phases of the lens slices and of the propagators are sampled on the transverse and angular wavenumber grids. A smooth phase, like the propagators', aliases in `e^(i f)` once it changes by more than `pi` between neighboring samples; the phase of a lens slice is a step at the edge of the slice instead. `experiment(**overrides)` returns its parameter set, the lengths of the paper's experiment scaled by `LENGTH_SCALE` (1.493) with overrides, and `explore(params, lens_slice)` prints the largest phase change between neighbors and plots the phase, `e^(i phase)` and their spectra of a lens slice and of the free propagations over a slice, the lens and the focal length. It compares the parameters of the paper with fewer, thicker slices of a stronger lens (`refractive_index=1.5`, 100 slices, `max_delta=0.1`) and unscaled phases.

## Running on the cluster

The SLURM jobs in `scripts/cluster/` at the repository root run `simulate.py` on a cluster with a clone of the repository. Both change from the submission directory to the repository root, two levels up, activate the workspace's `.venv` and run the script with `uv run --no-sync`. The environment holds the packages only, without the dependency groups of development (`uv sync --all-packages --no-default-groups --inexact`, which keeps anything else installed). They request 8 CPUs, 50 GB of memory and up to 120 hours on the partitions `long`, `standard`, `gpu` and `gpu-test`, and write the log to `.result/slurm-<job id>.out` relative to the submission directory, which SLURM does not create.

The single run synchronizes the environment itself and is submitted from `scripts/cluster/`. The tasks of the batch run concurrently and would race on the same `.venv`, so `scripts/cluster/submit-batch.sh` synchronizes it once and then submits the batch:

```sh
cd scripts/cluster
mkdir -p .result
sbatch run.slurm        # the single run of the paper
cd ../..
bash scripts/cluster/submit-batch.sh  # the batch over max_delta
```

| Job | Runs | Stores into |
| --- | --- | --- |
| `run.slurm` | One run with `--max-delta=0.01 --reverse-order --fresnel-approximation`, the sample-based propagator and the default 7 qubits, 10000 slices and 300 steps. | `apps/wave_optics_propagation/.result/<uuid>/`, the default results directory. |
| `batch-run.slurm` | A job array of 30 tasks; task `i` runs with `--reverse-order` and the `i`-th of 30 equally spaced `max_delta` from 0.001 to 0.3, the sample-based propagator and the spherical surface. | `apps/wave_optics_propagation/.result/batch/<uuid>/`, by `--results-dir`, where `batch_analysis.py` reads them. |

`--results-dir` is resolved against the working directory of the script, the repository root in the jobs. The results of the cluster are then copied into the app's `.result/` of the machine running the analyses.

## Results

`.result/` in the app's directory, not committed, holds the runs, one folder per run:

| Folder | Holds |
| --- | --- |
| `.result/<timestamp>/` | The local runs of December 2025 and January 2026, e.g. `2026-01-13_13-20-27`. |
| `.result/<uuid>/` | The runs of the cluster and of the current `simulate.py`, named by their uuid. |
| `.result/batch/<uuid>/` | The batch over `max_delta`. |
| `.result/legacy/` | The runs of an early notebook, stored before the lens model was saved with them. |

A run folder holds `initial_parameters.json`, the parameters with the derived `lens_diameter` and `lens_thickness` and the scalar results (`total_lenses_simulated`, `total_probability_of_success`, `success_probabilities`), and `results.npz`, the snapshots by name. `qiu_classical_simulation.wave_optics.storage` reads them: `load_experiment(folder)` returns the parameters and the result, `load_parameters(folder)` only the parameters, and `run_folders(results_dir)` the run folders of a directory, sorted by name. `save_experiment` never overwrites a stored run: it raises a `FileExistsError` for a uuid stored already, e.g. of parameters copied with `dataclasses.replace`, which keeps the uuid; such a copy needs a new one, `replace(parameters, ..., uuid=uuid.uuid4().hex)`. `.output/`, not committed either, holds the figures of the analyses.

### Reading older runs

The storage reads the runs of all former versions:

- Snapshots stored as column vectors are flattened.
- The former keys `timestamp` and `reverse_order` are read as `experiment_datetime` and `lens_reverse_order` (`LEGACY_KEYS` of `qiu_classical_simulation.wave_optics.parameters`).
- A missing `direct_propagator` is `True`, which the former code always used, and a missing `uuid` is empty (`LEGACY_DEFAULTS`).
- Other missing parameters raise a `KeyError` naming them, and are given explicitly: `load_experiment(folder, defaults={...})`.

The local runs of December 2025 and January 2026 read without defaults. The runs in `.result/legacy/` lack `lens_reverse_order`, `fresnel_approximation` and `scale_down_phases`, which have to be given as defaults with the values the notebook used.

## Changes to the former analyses

The simulation and the analyses were reworked into the shared [qiu-classical-simulation](../qiu-classical-simulation/index.md), with these differences to the former code:

- The reverse order passes the slices from the plane side, `N-1, ..., 0`; the former code passed them as `0, N-1, ..., 1`. The difference in the final field of the experiment of the paper is an infidelity of about `2e-6`.
- The batch analysis reads the success probability at the analyzed snapshot; the former one read it one free space step earlier.
- Each run gets its own uuid and time; formerly, all runs of one process shared them.

## Tests

The tests check that the backend's phase protocol matches the closed form of the post-selected cycles, state and success probability, for both state preparation methods, that its direct propagator matches the exact one, and that a small simulated experiment approaches the exact one for small `max_delta`. From the repository root:

```sh
uv run pytest apps/wave_optics_propagation
```
