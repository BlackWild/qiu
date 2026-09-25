# wave_optics_propagation_qutip

The lens experiments of the paper, simulated with QuTiP instead of Qiskit: a Gaussian beam passes a plano-convex lens, sliced along the optical axis into thin transparent plates, and then free space behind it. The sample-based phase protocol acts on QuTiP kets of an ancilla register and the transverse field, post-selected on its success; the direct propagator is an operator conjugated by the discrete Fourier transform.

The experiment itself, i.e. its parameters, its simulation loop, its storage and its analysis, lives in [qiu-classical-simulation](../qiu-classical-simulation/) and is shared with the Qiskit simulation, [wave_optics_propagation](../wave_optics_propagation/), which also documents the experiment, the scripts and the cluster runs in depth. This app adds the QuTiP backend, [`QutipBackend`][wave_optics_propagation_qutip.backend.QutipBackend], and two section-based (`# %%`) scripts in `scripts/`: the simulation and the analysis of a single run.

## Installation

The app is a member of the uv workspace and is not published. It is installed with all other packages, from the repository root:

```sh
uv sync --all-packages
```

Its scripts are run with the workspace environment, e.g. `uv run python apps/wave_optics_propagation_qutip/scripts/simulate.py --help`.

## Quick start

A small version of the experiment of the paper, 32 transverse samples, 10 lens slices and 3 free space steps, simulated with the QuTiP backend and compared with the exact application of the same phases:

```python
from pathlib import Path

import numpy as np
from qiu_classical_simulation.wave_optics.cli import parse_parameters
from qiu_classical_simulation.wave_optics.simulation import ExactBackend, simulate
from wave_optics_propagation_qutip.backend import QutipBackend

parameters, _ = parse_parameters(
    "Simulate the lens experiment with QuTiP.",
    Path(".result"),
    [
        "--max-delta=0.05",
        "--num-qubits=5",
        "--lens-slices=10",
        "--steps-after-lens=3",
        "--reverse-order",
        "--direct-propagator",
    ],
)
result = simulate(parameters, QutipBackend())
exact = simulate(parameters, ExactBackend())

fidelity = abs(np.vdot(result.snapshots["final"], exact.snapshots["final"])) ** 2
assert fidelity > 0.999
assert 0 < result.total_probability_of_success < 1
```

The protocol reproduces the exact field with a fidelity above 0.999, and all its cycles succeed with a probability of about 0.48, the same as with the Qiskit backend.

## Where next

- The [User Guide](user-guide.md) describes the experiment, the QuTiP backend and its operators, the scripts and the stored runs, including those of December 2025.
- The [Examples](examples.md) simulate small experiments, apply a cycle of the protocol by hand, compare the field behind the lens with the classical numerics, and read a run stored by a former version.
- The [API Reference](reference/wave_optics_propagation_qutip/index.md) documents the backend.
