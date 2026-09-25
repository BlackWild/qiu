# Examples

The examples simulate small versions of the experiment of the paper, 16 or 32 transverse samples and 10 to 100 lens slices instead of 128 samples and 10000 slices, so that each runs in a few seconds; their fields are coarse, and `validity_problems()` reports the spacing above the wavelength.

## Simulating a small lens experiment

The parameters are given directly as an `ExperimentParameters`, those of the paper with 5 qubits, 10 slices and 3 free space steps behind the lens. The Qiskit backend applies all phases with the sample-based protocol, the free propagation included; `ExactBackend` applies the same phases exactly, and the classical numerics compute the exact field behind the lens in a single propagation.

```python
import numpy as np
from python_wave_optics.classical_numerics import classical_numerics_simulation
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.simulation import ExactBackend, simulate
from wave_optics_propagation.backend import QiskitBackend

parameters = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-6,
    focal_length=200e-6,
    refractive_index=1.25,
    propagation_after_lens=300e-6,
    transverse_length=100e-6,
    num_of_steps_after_lens=3,
    lens_slices=10,
    num_qubits=5,
    max_delta=0.05,
    lens_reverse_order=True,
    fresnel_approximation=False,
    scale_down_phases=True,
    direct_propagator=False,
)
result = simulate(parameters, QiskitBackend())
exact = simulate(parameters, ExactBackend())

# a snapshot after each slice and step, and the entering, behind-lens and final ones
assert len(result.snapshots) == 10 + 3 + 3
assert list(result.snapshots) == list(exact.snapshots)

final = result.snapshots["final"]
fidelity = abs(np.vdot(final, exact.snapshots["final"])) ** 2
assert 0.99 < fidelity < 1

classical = classical_numerics_simulation(parameters, parameters.propagation_after_lens)
assert abs(abs(np.vdot(exact.snapshots["final"], classical)) - 1) < 1e-9

probability = result.total_probability_of_success
assert probability == result.success_probability("final")
assert 0.4 < probability < 0.5
```

The simulated final field has a fidelity of about 0.997 to the exact one, whose field matches the classical numerics up to rounding; all cycles of the protocol succeed with a probability of about 0.46.

## The accuracy and the success probability over max_delta

The batch analysis of the paper compares, for runs over `max_delta`, the field two thirds of the way behind the lens with the classical numerics, and fits the fidelity by `1 + a max_delta**2` and the success probability by `exp(a max_delta)`. The same on a small experiment of 4 qubits, built from the command line of the scripts with argument overrides:

```python
from pathlib import Path

import numpy as np
from python_wave_optics.classical_numerics import classical_numerics_simulation
from python_wave_optics.cli import argument_parser, parameters_from_arguments
from python_wave_optics.result import free_space_snapshot_name
from python_wave_optics.simulation import simulate
from wave_optics_propagation.backend import QiskitBackend

parser = argument_parser("Simulate the lens experiment with Qiskit.", Path(".result"))
max_deltas = np.array([0.2, 0.1, 0.05])
infidelities, success_probabilities = [], []
for max_delta in max_deltas:
    arguments = parser.parse_args(
        [
            f"--max-delta={max_delta}",
            "--num-qubits=4",
            "--lens-slices=10",
            "--steps-after-lens=3",
            "--reverse-order",
        ]
    )
    parameters = parameters_from_arguments(arguments)
    result = simulate(parameters, QiskitBackend())

    reference_step = parameters.num_of_steps_after_lens * 2 // 3 + 1
    reference_snapshot = free_space_snapshot_name(reference_step)
    state = result.snapshots[reference_snapshot]
    classical = classical_numerics_simulation(
        parameters, parameters.step_size_after_lens * reference_step
    )
    infidelities.append(1 - abs(np.vdot(state, classical)) ** 2)
    success_probabilities.append(result.success_probability(reference_snapshot))

# the fidelity decreases quadratically, the success probability exponentially
a_fidelity = -np.mean(np.array(infidelities) / max_deltas**2)
fitted_fidelities = 1 + a_fidelity * max_deltas**2
assert np.allclose(1 - np.array(infidelities), fitted_fidelities, atol=5e-4)
a_success, log_offset = np.polyfit(max_deltas, np.log(success_probabilities), 1)
assert a_success < 0 and abs(log_offset) < 0.1
assert np.all(np.diff(success_probabilities) > 0)
```

Halving `max_delta` reduces the infidelity by a factor of about 4, from about `5e-3` at 0.2 to `3e-4` at 0.05, while the success probability grows from about 0.06 to 0.49, following `exp(a max_delta)` with `a` about -14.

## Storing and loading a run

`simulate.py` stores each run with `save_experiment` in `<results-dir>/<uuid>/`, where the analyses load it back with `load_experiment`. The same in a temporary directory, for a run with the direct propagator:

```python
import tempfile
from pathlib import Path

import numpy as np
from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import simulate
from python_wave_optics.storage import (
    load_experiment,
    load_stored_values,
    run_folders,
    save_experiment,
)
from wave_optics_propagation.backend import QiskitBackend

with tempfile.TemporaryDirectory() as results_dir:
    parameters, _ = parse_parameters(
        "",
        Path(results_dir),
        [
            "--max-delta=0.1",
            "--num-qubits=4",
            "--lens-slices=10",
            "--steps-after-lens=3",
            "--direct-propagator",
        ],
    )
    result = simulate(parameters, QiskitBackend())
    folder = save_experiment(results_dir, parameters, result)

    assert folder == Path(results_dir) / parameters.uuid
    assert run_folders(results_dir) == [folder]
    assert sorted(path.name for path in folder.iterdir()) == [
        "initial_parameters.json",
        "results.npz",
    ]

    loaded_parameters, loaded_result = load_experiment(folder)
    assert loaded_parameters == parameters
    assert list(loaded_result.snapshots) == list(result.snapshots)
    for name, state in result.snapshots.items():
        assert np.array_equal(loaded_result.snapshots[name], state)
    assert loaded_result.success_probabilities == result.success_probabilities

    stored = load_stored_values(folder)
    assert stored["lens_thickness"] == parameters.lens_thickness
    assert stored["total_lenses_simulated"] == result.total_lenses_simulated
```

The loaded parameters, uuid and time included, equal the stored ones, and the snapshots are identical; the JSON file also holds the derived lens geometry and the scalar results.

## The beam waist along the propagation

`forward_analysis.py` compares the beam waist of every snapshot with the one behind an ideal thin lens of the same focal length at the principal plane. The lens has to be sliced finely enough to focus the beam: with 10 slices, the stepped phase barely focuses it; with 100 slices, of which only the 50 with a phase are simulated in the Fresnel approximation, the simulated focus coincides with the thin lens'.

```python
from pathlib import Path

import numpy as np
from python_wave_optics.analysis import (
    beam_waist,
    principal_plane_position,
    propagation_distances,
    thin_lens_reference_states,
)
from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import simulate
from wave_optics_propagation.backend import QiskitBackend

parameters, _ = parse_parameters(
    "",
    Path(".result"),
    [
        "--max-delta=0.05",
        "--num-qubits=5",
        "--lens-slices=100",
        "--steps-after-lens=12",
        "--reverse-order",
        "--fresnel-approximation",
        "--direct-propagator",
    ],
)
result = simulate(parameters, QiskitBackend())
assert result.total_lenses_simulated == 50

states = result.lens_states(parameters.lens_slices) + result.free_space_states(
    parameters.num_of_steps_after_lens
)
distances = np.array(propagation_distances(parameters))
x_values = parameters.x_axis.values
beam_waists = np.array([beam_waist(state, x_values) for state in states])
thin_lens_waists = np.array(
    [beam_waist(state, x_values) for state in thin_lens_reference_states(parameters)]
)

focal_point = distances[np.argmin(beam_waists)]
thin_lens_focal_point = distances[np.argmin(thin_lens_waists)]
assert principal_plane_position(parameters) == parameters.lens_thickness
assert abs(focal_point - thin_lens_focal_point) <= parameters.step_size_after_lens
assert abs(thin_lens_focal_point - (parameters.lens_thickness + 200e-6)) < 1e-9
assert beam_waists.min() < 0.6 * parameters.gaussian_beam_waist
```

The beam narrows from its waist of 21.2 um to about 10 um at 250 um behind the entrance, the principal plane at the lens' convex vertex, 50 um, plus the focal length of 200 um, where the thin lens focuses it too. On 32 samples, the focus stays wider than the thin lens' of about 3 um.

## Running the simulation script

`simulate.py` runs the same experiments from the command line, from any working directory, and stores the run in the app's `.result/` or in `--results-dir`; the defaults are the experiment of the paper. A small run, and the single run of `cluster/run.slurm`:

```sh
uv run python apps/wave_optics_propagation/scripts/simulate.py --help
uv run python apps/wave_optics_propagation/scripts/simulate.py --max-delta=0.1 \
    --num-qubits=5 --lens-slices=100 --steps-after-lens=12 --reverse-order \
    --fresnel-approximation --direct-propagator --results-dir=/tmp/lens-runs
uv run python apps/wave_optics_propagation/scripts/simulate.py --max-delta=0.01 \
    --reverse-order --fresnel-approximation
```

Each run prints its parameters, warnings about their validity, a progress bar per loop, the folder it was stored in and its total probability of success.
