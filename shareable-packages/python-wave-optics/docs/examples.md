# Examples

## A weak lens against the thin lens analytics

A thin, weakly curved lens should focus the beam as an ideal thin lens of the same focal length does. This example simulates such a lens exactly, checks the snapshots against the split-step classical numerics, and compares the beam waist along the propagation with the one behind the thin lens at the principal plane.

```python
import matplotlib.pyplot as plt
import numpy as np
from python_wave_optics.analysis import (
    beam_waist,
    propagation_distances,
    thin_lens_reference_states,
)
from python_wave_optics.classical_numerics import classical_numerics_simulation
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.simulation import ExactBackend, simulate
from python_wave_optics.visualization import plot_wavefunction

parameters = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-6,
    focal_length=1e-3,
    refractive_index=1.5,
    propagation_after_lens=1.5e-3,
    transverse_length=100e-6,
    num_of_steps_after_lens=30,
    lens_slices=50,
    num_qubits=7,
    max_delta=0.1,
    lens_reverse_order=False,
    fresnel_approximation=False,
    scale_down_phases=True,
    direct_propagator=True,
)
assert parameters.is_valid()
assert parameters.lens_thickness < 3e-6  # a thin lens of R = 500 um

result = simulate(parameters, ExactBackend())
assert result.total_probability_of_success == 1.0

# the exact backend is the split-step numerics
final = result.snapshots["final"]
reference = classical_numerics_simulation(parameters, parameters.propagation_after_lens)
assert abs(np.vdot(reference, final)) > 1 - 1e-9

# the beam waists behind the lens, simulated and behind the ideal thin lens
x = parameters.x_axis.values
steps = parameters.num_of_steps_after_lens
simulated = [beam_waist(state, x) for state in result.free_space_states(steps)]
thin_lens = [beam_waist(state, x) for state in thin_lens_reference_states(parameters)]
thin_lens = thin_lens[parameters.lens_slices :]
assert np.allclose(simulated, thin_lens, rtol=0.03)

distances = np.array(propagation_distances(parameters)[parameters.lens_slices :])
focus = distances[np.argmin(simulated)]
assert focus == distances[np.argmin(thin_lens)]
assert min(simulated) < 0.65 * parameters.gaussian_beam_waist

fig, (magnitude_axes, phase_axes) = plot_wavefunction(result.snapshots["after_lens"])
plt.close(fig)
```

The simulated waists agree with the thin lens ones within 3 % all along the 1.5 mm behind the lens, and both focus the beam from about 21 um to about 12 um at the same step, about 0.65 mm behind the lens: before the focal plane, as for any Gaussian beam of a finite Rayleigh range, here 1.4 mm. For the strongly curved hemisphere of the paper (`focal_length=200e-6`, `refractive_index=1.25`), the same comparison shows the focus of the thick lens moving towards the lens and widening.

## A backend of the ideal phase protocol

A backend decides how the phases are applied. This one applies the sample-based phase protocol in its closed form, with the functions of `phase_protocol`, so that its errors and its probability of success can be studied without a quantum simulation.

```python
import dataclasses

import numpy as np
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.phase_protocol import decompose, ideal_cycles, slice_phase
from python_wave_optics.simulation import (
    ExactBackend,
    PropagationBackend,
    from_angular_spectrum,
    simulate,
    to_angular_spectrum,
)


class IdealProtocolBackend(PropagationBackend):
    """Applies the lens phases by ideal cycles of the phase protocol."""

    def sample_based_phase(self, signal, max_delta):
        alpha, phi = decompose(signal)  # prepared once per signal
        deltas = slice_phase(alpha, max_delta)
        return lambda state: ideal_cycles(state, phi, deltas)

    def direct_momentum_phase(self, signal):
        factors = np.exp(1j * signal.data)
        return lambda state: (
            from_angular_spectrum(factors * to_angular_spectrum(state)),
            1.0,
        )


parameters = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-6,
    focal_length=1e-3,
    refractive_index=1.5,
    propagation_after_lens=1.5e-3,
    transverse_length=100e-6,
    num_of_steps_after_lens=10,
    lens_slices=20,
    num_qubits=7,
    max_delta=0.1,
    lens_reverse_order=False,
    fresnel_approximation=False,
    scale_down_phases=True,
    direct_propagator=True,
)
exact = simulate(parameters, ExactBackend()).snapshots["final"]

runs = {}
for max_delta in [0.1, 0.01]:
    variant = dataclasses.replace(parameters, max_delta=max_delta)
    result = simulate(variant, IdealProtocolBackend())
    fidelity = abs(np.vdot(exact, result.snapshots["final"])) ** 2
    runs[max_delta] = (fidelity, result.total_probability_of_success)

# smaller phases per cycle: more accurate, and more likely to succeed
assert runs[0.01][0] > runs[0.1][0] > 0.99
assert runs[0.01][0] > 0.9999
assert 0 < runs[0.1][1] < runs[0.01][1] < 1

# with the direct propagator, only the lens slices can fail
assert result.success_probability("after_lens") == result.total_probability_of_success
```

Both runs reproduce the exact field closely, the one of `max_delta = 0.01` with a fidelity above `0.9999`. Their probabilities of success, the product over all cycles, are below 1 and grow as `max_delta` shrinks, while the number of cycles grows as `1 / max_delta`. The quantum backends of the apps implement the same two methods with Qiskit circuits and QuTiP operators.

## Storing and loading runs

Each run is stored in its own folder, named by the ID of its parameters, and loads back as it was. Runs of older versions of the apps may lack parameters; they are given as defaults when loading.

```python
import dataclasses
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.simulation import ExactBackend, simulate
from python_wave_optics.storage import (
    PARAMETERS_FILENAME,
    load_experiment,
    run_folders,
    save_experiment,
)

parameters = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-6,
    focal_length=200e-6,
    refractive_index=1.25,
    propagation_after_lens=300e-6,
    transverse_length=100e-6,
    num_of_steps_after_lens=4,
    lens_slices=12,
    num_qubits=5,
    max_delta=0.1,
    lens_reverse_order=False,
    fresnel_approximation=False,
    scale_down_phases=True,
    direct_propagator=True,
)
# a variant of the experiment, with its own ID so that it gets its own folder
reverse = dataclasses.replace(parameters, lens_reverse_order=True, uuid=uuid.uuid4().hex)

with tempfile.TemporaryDirectory() as results_dir:
    for run in [parameters, reverse]:
        save_experiment(results_dir, run, simulate(run, ExactBackend()))

    folders = run_folders(results_dir)
    assert sorted(folder.name for folder in folders) == sorted([parameters.uuid, reverse.uuid])

    folder = Path(results_dir) / parameters.uuid
    loaded_parameters, loaded_result = load_experiment(folder)
    assert loaded_parameters == parameters
    assert list(loaded_result.snapshots)[0] == "step_0"
    assert np.allclose(loaded_result.snapshots["step_0"], parameters.initial_state)

    # a run of an older version: former names, and a parameter it did not store
    values = json.loads((folder / PARAMETERS_FILENAME).read_text())
    values["reverse_order"] = values.pop("lens_reverse_order")
    del values["fresnel_approximation"], values["direct_propagator"], values["uuid"]
    (folder / PARAMETERS_FILENAME).write_text(json.dumps(values))

    try:
        load_experiment(folder)
    except KeyError as error:
        assert "fresnel_approximation" in str(error)
    else:
        raise AssertionError("the missing parameter must be given")

    old, _ = load_experiment(folder, defaults={"fresnel_approximation": False})
    assert old.lens_reverse_order is False and old.direct_propagator is True
    assert old.uuid == ""
```

The stored run loads as equal parameters and the same snapshots. The rewritten file of the older format loads once its missing `fresnel_approximation` is given, with the former key `reverse_order` renamed and `direct_propagator` filled in as the former code always used it.

## A Gaussian beam through free space and a thin lens

The beam waist of a numerically propagated field can be compared with the analytic Gaussian beam of `gaussian_beam`, e.g. to check that the window and the sampling are adequate before a long run.

```python
import numpy as np
from python_wave_optics.analysis import beam_waist
from python_wave_optics.classical_numerics import (
    propagate_exactly,
    thin_lens_simulation,
)
from python_wave_optics.gaussian_beam import (
    beam_after_free_space,
    beam_after_thin_lens,
    gaussian_signal,
    rayleigh_range,
)
from python_wave_optics.parameters import ExperimentParameters

parameters = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-6,
    focal_length=1e-3,
    refractive_index=1.5,
    propagation_after_lens=1.5e-3,
    transverse_length=100e-6,
    num_of_steps_after_lens=30,
    lens_slices=50,
    num_qubits=7,
    max_delta=0.1,
    lens_reverse_order=False,
    fresnel_approximation=False,
    scale_down_phases=True,
    direct_propagator=True,
)
x = parameters.x_axis.values
waist = parameters.gaussian_beam_waist

# the sampled beam has the waist of the formulas
assert np.isclose(beam_waist(parameters.initial_state, x), waist, rtol=1e-3)
profile = gaussian_signal(parameters.x_axis, waist, parameters.gaussian_mean)
assert np.allclose(profile.to_signal().normalized_data, parameters.initial_state)

# free space: the numerical spreading follows w(z) = w0 sqrt(1 + (z / z_R)**2)
for distance in [0.5e-3, 1e-3, 1.5e-3]:
    field = propagate_exactly(parameters.initial_state, parameters, distance)
    _, expected = beam_after_free_space(waist, distance, 1e-6, refractive_index=1)
    z_r = rayleigh_range(waist, 1e-6)
    assert np.isclose(expected, waist * np.sqrt(1 + (distance / z_r) ** 2))
    assert np.isclose(beam_waist(field, x), expected, rtol=0.01)

# behind a thin lens: the focus of a beam with its waist at the lens
f = parameters.focal_length
focus = f / (1 + (f / z_r) ** 2)
_, focused = beam_after_thin_lens(waist, focus, f, 1e-6, refractive_index=1)
assert np.isclose(focused, waist / np.sqrt(1 + (z_r / f) ** 2))
profile = thin_lens_simulation(parameters, 0.0, focus)
assert np.isclose(beam_waist(profile, x), focused, rtol=1e-3)
```

The numerical free propagation follows the analytic beam radius within 1 % up to 1.5 mm, where the beam has spread from 21 um to about 31 um, still within the window of 100 um. Behind a thin lens, the waist lies at `f / (1 + (f / z_R)**2)`, about 0.67 mm, before the focal plane, with the radius `w0 / sqrt(1 + (z_R / f)**2)`; `thin_lens_simulation` samples this profile on the grid of the experiment.
