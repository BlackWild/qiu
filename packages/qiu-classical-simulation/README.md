# Qiu Classical Simulation

[![PyPI](https://img.shields.io/pypi/v/qiu-classical-simulation)](https://pypi.org/project/qiu-classical-simulation/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiu-classical-simulation/)

Tools for classical simulations. Its subpackage `wave_optics` simulates the paraxial wave optics of a Gaussian beam through a plano-convex lens, sliced along the optical axis, and through free space behind it. It holds everything about this lens experiment that does not depend on how the phases are applied, so that the simulations with Qiskit (`apps/wave_optics_propagation`) and QuTiP (`apps/wave_optics_propagation_qutip`) share it. It depends on NumPy, Matplotlib and [`qiu-signals`](https://github.com/BlackWild/qiu/blob/master/packages/qiu-signals/README.md).

## Installation

```sh
pip install qiu-classical-simulation
```

## Modules

The modules of the subpackage `wave_optics`:

- `gaussian_beam`: Gaussian beams through ABCD systems (`propagation_matrix`, `thin_lens_matrix`, `beam_after_system`), the beam after free space or a thin lens (`beam_after_free_space`, `beam_after_thin_lens`), and the Gaussian field on an axis (`gaussian_signal`).
- `elements`: The phase signals of the thin elements: the transverse radius of a plano-convex lens at a depth (`convex_planar_lens_radius`), a transparent plate (`transparent_plate_phase`), and the paraxial free propagation `-k**2 dz / (2 k0)` of the angular spectrum (`free_space_propagator_phase`).
- `parameters`: `ExperimentParameters`, the given parameters of an experiment, from which the lens geometry, the grid, the initial beam and the phase signals of the lens slices derive. `ordered_lens_signals` gives the slices in the order the beam passes them, from the plane side for `lens_reverse_order`.
- `simulation`: `simulate(parameters, backend)` runs the experiment, taking a snapshot after each lens slice and free space step. A `PropagationBackend` applies the phases: `sample_based_phase(signal, max_delta)` with the sample-based phase protocol, post-selected on its success, and `direct_momentum_phase(signal)` on the angular spectrum. `ExactBackend` applies them exactly.
- `phase_protocol`: The arithmetic of the protocol, independent of its implementation: `decompose` (`f = alpha |phi|**2`), `slice_phase` and the closed form of the cycles, `ideal_cycles`.
- `result` and `storage`: `ExperimentResult`, the named snapshots and the cumulative success probabilities (`success_probability(snapshot)`), stored per run in `<results_dir>/<uuid>/` as `initial_parameters.json` and `results.npz` (`save_experiment`, `load_experiment`, `run_folders`).
- `classical_numerics`: The references: the exact split-step field behind the lens (`classical_numerics_simulation`) and the analytic profile behind an ideal thin lens (`thin_lens_simulation`).
- `analysis`: The beam waist of a field, the principal plane, the propagation distances of the snapshots, the thin lens references along them, and the lens surface for plots.
- `visualization`: `plot_wavefunction`, the magnitude and phase of a field side by side.
- `cli`: The command line of the simulation scripts, with the experiment of the paper as defaults.

## Stored runs

The storage reads the results of all former versions of the apps:

- Snapshots stored as column vectors are flattened.
- The former keys `timestamp` and `reverse_order` are read as `experiment_datetime` and `lens_reverse_order`.
- A missing `direct_propagator` is `True`, which the former code always used.
- Other missing parameters, e.g. `fresnel_approximation` of the runs of December 2025, are given explicitly: `load_experiment(folder, defaults={...})`.

## Usage

```python
from qiu_classical_simulation.wave_optics.cli import parse_parameters
from qiu_classical_simulation.wave_optics.classical_numerics import (
    classical_numerics_simulation,
)
from qiu_classical_simulation.wave_optics.simulation import ExactBackend, simulate

parameters, _ = parse_parameters(
    "", ".result", ["--max-delta=0.1", "--num-qubits=5", "--lens-slices=20"]
)
result = simulate(parameters, ExactBackend())

behind_lens = parameters.step_size_after_lens * parameters.num_of_steps_after_lens
reference = classical_numerics_simulation(parameters, behind_lens)
assert abs(abs(result.snapshots["final"] @ reference.conj()) - 1) < 1e-9
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiu-classical-simulation/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiu-classical-simulation/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiu-classical-simulation
```
