# Examples

The examples simulate small versions of the experiment of the paper, 32 transverse samples and 10 to 100 lens slices instead of 128 samples and 10000 slices, so that each runs in a few seconds; their fields are coarse, and `validity_problems()` reports the spacing above the wavelength.

## Simulating a small lens experiment

The parameters are given directly as an `ExperimentParameters`, those of the paper with 5 qubits, 10 slices and 3 free space steps behind the lens. The experiment is simulated with the direct and with the sample-based propagator and compared with `ExactBackend`, which applies the same phases exactly.

```python
import dataclasses

import numpy as np
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.simulation import ExactBackend, simulate
from wave_optics_propagation_qutip.backend import QutipBackend

direct = ExperimentParameters(
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
    direct_propagator=True,
)
sample_based = dataclasses.replace(direct, direct_propagator=False)

fidelities, probabilities = [], []
for parameters in [direct, sample_based]:
    result = simulate(parameters, QutipBackend())
    exact = simulate(parameters, ExactBackend())
    assert list(result.snapshots) == list(exact.snapshots)
    final, exact_final = result.snapshots["final"], exact.snapshots["final"]
    fidelities.append(abs(np.vdot(final, exact_final)) ** 2)
    probabilities.append(result.total_probability_of_success)

# the direct propagator is exact, the sample-based one adds errors and cycles
assert fidelities[0] > fidelities[1] > 0.99
assert probabilities[0] > probabilities[1] > 0.4
```

With the direct propagator, only the lens slices are approximated, with an infidelity of about `2.5e-4`; the sample-based propagator raises it to about `2.7e-3` and lowers the probability that all cycles succeed from about 0.48 to 0.46.

## A cycle of the protocol by hand

The backend's protocol is built from two operators: the partial phase on the ket `|phi> (x) |psi>` of the ancilla and the field, and the projection of the ancilla onto `<phi|`. One cycle by hand, compared with the closed form of `python_wave_optics.phase_protocol.ideal_cycles`, and all cycles of a signal by the backend, compared with `e^(i f) psi`:

```python
import numpy as np
import qutip as qt
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_wave_optics.phase_protocol import decompose, ideal_cycles, slice_phase
from wave_optics_propagation_qutip.backend import (
    QutipBackend,
    dft_operator,
    partial_phase_operator,
)

dimension = 8
axis = PositionAxis(dimension, 1.0, IndexOrdering.NATURAL)
signal = AlgebraicSignal(axis, lambda x: 0.1 * (1 + x**2))
alpha, amplitudes = decompose(signal)
deltas = slice_phase(alpha, max_delta=0.1)
assert np.allclose(alpha * amplitudes**2, signal.data)
assert len(deltas) == 148  # alpha = 14.8

rng = np.random.default_rng(0)
psi = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
psi /= np.linalg.norm(psi)

# one cycle: the partial phase, then the projection of the ancilla onto <phi|
phi = qt.Qobj(amplitudes.astype(np.complex128))
partial_phase = partial_phase_operator(deltas[0], dimension)
projection = qt.tensor(phi.dag(), qt.qeye(dimension))
successful = (projection @ partial_phase @ qt.tensor(phi, qt.Qobj(psi))).full().ravel()
success = np.vdot(successful, successful).real

expected, expected_success = ideal_cycles(psi, amplitudes, deltas[:1])
assert np.allclose(successful / np.sqrt(success), expected)
assert np.isclose(success, expected_success)

# all cycles, by the backend
state, probability = QutipBackend().sample_based_phase(signal, 0.1)(psi)
expected, expected_probability = ideal_cycles(psi, amplitudes, deltas)
assert np.allclose(state, expected)
assert np.isclose(probability, expected_probability)
assert abs(np.vdot(state, np.exp(1j * signal.data) * psi)) ** 2 > 0.99

# the DFT of the direct propagator is numpy's orthonormal FFT
assert np.allclose(dft_operator(dimension).full() @ psi, np.fft.fft(psi, norm="ortho"))
```

A single cycle succeeds with a probability of about 0.999 and matches the closed form; the 148 cycles of the signal succeed together with a probability of about 0.84 and apply `e^(i f)` with an infidelity of about `2.6e-3`.

## The field behind the lens against the classical numerics

`analysis.py` compares the snapshots behind the lens with the thin lens and with the classical numerics at the same distance: the exact split-step field of the same slicing. With 100 slices in the Fresnel approximation, of which the 50 with a phase are simulated, and the direct propagator, which propagates exactly, the fidelity to the classical numerics is the one reached behind the lens at every step, and the simulated beam focuses where the classical one does.

```python
from pathlib import Path

import numpy as np
from python_wave_optics.analysis import beam_waist, thin_lens_reference_states
from python_wave_optics.classical_numerics import classical_numerics_simulation
from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import simulate
from wave_optics_propagation_qutip.backend import QutipBackend

params, _ = parse_parameters(
    "",
    Path(".result"),
    [
        "--max-delta=0.02",
        "--num-qubits=5",
        "--lens-slices=100",
        "--steps-after-lens=12",
        "--reverse-order",
        "--fresnel-approximation",
        "--direct-propagator",
    ],
)
results = simulate(params, QutipBackend())
assert results.total_lenses_simulated == 50

x_values = params.x_axis.values
free_space_states = results.free_space_states(params.num_of_steps_after_lens)
thin_lens_states = thin_lens_reference_states(params)[params.lens_slices :]
infidelities, waists, classical_waists, thin_lens_waists = [], [], [], []
for step, state in enumerate(free_space_states):
    distance = params.step_size_after_lens * (step + 1)
    classical = classical_numerics_simulation(params, distance)
    infidelities.append(1 - abs(np.vdot(state, classical)) ** 2)
    waists.append(beam_waist(state, x_values))
    classical_waists.append(beam_waist(classical, x_values))
    thin_lens_waists.append(beam_waist(thin_lens_states[step], x_values))

assert np.allclose(infidelities, infidelities[0]) and infidelities[0] < 1e-3
assert np.argmin(waists) == np.argmin(classical_waists)
assert np.allclose(waists, classical_waists, rtol=0.1)
assert min(waists) < 0.5 * params.gaussian_beam_waist
assert min(thin_lens_waists) < 0.5 * min(waists)
```

The infidelity to the classical numerics stays at about `5e-4` over all 12 steps. The simulated and the classical beam waists agree within 10 % and are both smallest 175 um behind the lens, at about 9 um; the ideal thin lens, 200 um behind the principal plane, focuses more tightly, to about 3 um on this grid.

## Reading a run stored by a former version

The runs of December 2025 were stored with the former keys `timestamp` and `reverse_order`, snapshots as column vectors, and without the lens model and the propagator. A run stored now, rewritten into that form in a temporary directory, reads back the same way, with the parameters the former script used given as defaults:

```python
import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import simulate
from python_wave_optics.storage import load_experiment, save_experiment
from wave_optics_propagation_qutip.backend import QutipBackend

LEGACY_DEFAULTS = {"fresnel_approximation": True, "scale_down_phases": True}

with tempfile.TemporaryDirectory() as results_dir:
    parameters, _ = parse_parameters(
        "",
        Path(results_dir),
        [
            "--max-delta=0.1",
            "--num-qubits=4",
            "--lens-slices=10",
            "--steps-after-lens=3",
            "--reverse-order",
            "--fresnel-approximation",
            "--direct-propagator",
        ],
    )
    result = simulate(parameters, QutipBackend())
    folder = save_experiment(results_dir, parameters, result)

    # the former form: renamed keys, no lens model, propagator or uuid, column vectors
    values = json.loads((folder / "initial_parameters.json").read_text())
    values["timestamp"] = values.pop("experiment_datetime")
    values["reverse_order"] = values.pop("lens_reverse_order")
    for key in ["fresnel_approximation", "scale_down_phases", "direct_propagator"]:
        del values[key]
    del values["uuid"]
    (folder / "initial_parameters.json").write_text(json.dumps(values))
    columns = {name: state.reshape(-1, 1) for name, state in result.snapshots.items()}
    np.savez(folder / "results.npz", **columns)

    try:
        load_experiment(folder)
    except KeyError as error:
        assert "fresnel_approximation" in str(error)
    else:
        raise AssertionError("The lens model must be given as defaults.")

    loaded_parameters, loaded_result = load_experiment(folder, defaults=LEGACY_DEFAULTS)
    assert loaded_parameters == dataclasses.replace(parameters, uuid="")
    assert loaded_result.snapshots["final"].shape == (parameters.dimension,)
    assert np.array_equal(loaded_result.snapshots["final"], result.snapshots["final"])
```

Without the defaults, loading fails with a `KeyError` naming the missing parameters; with them, the parameters equal the stored ones but for the empty uuid, the missing `direct_propagator` reads as `True`, and the snapshots are flattened.

## Running the scripts

`simulate.py` runs the experiments from the command line, from any working directory, and stores the run in the app's `.result/` or in `--results-dir`; the defaults are the experiment of the paper. `analysis.py` then analyzes the run of its `EXPERIMENT_FOLDER`:

```sh
uv run python apps/wave_optics_propagation_qutip/scripts/simulate.py --help
uv run python apps/wave_optics_propagation_qutip/scripts/simulate.py --max-delta=0.1 \
    --num-qubits=5 --lens-slices=100 --steps-after-lens=12 --reverse-order \
    --fresnel-approximation --direct-propagator
uv run python apps/wave_optics_propagation_qutip/scripts/analysis.py
```

Each simulation prints its parameters, warnings about their validity, a progress bar per loop, the folder it was stored in and its total probability of success.
