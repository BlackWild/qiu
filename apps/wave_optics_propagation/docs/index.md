# wave_optics_propagation

The lens experiments of the paper, simulated with Qiskit: a Gaussian beam passes a plano-convex lens, sliced along the optical axis into thin transparent plates, and then free space behind it. The phase of each lens slice, and optionally of each free propagation, is applied with the sample-based phase protocol, simulated on statevectors and post-selected on its success; the result is compared with the classical numerics and with an ideal thin lens.

The experiment itself, i.e. its parameters, its simulation loop, its storage and its analysis, lives in [qiu-classical-simulation](../qiu-classical-simulation/index.md) and is shared with the QuTiP simulation, [wave_optics_propagation_qutip](../wave_optics_propagation_qutip/index.md). This app adds the Qiskit backend, [`QiskitBackend`][wave_optics_propagation.backend.QiskitBackend], built on the circuits of [qiu-quantum-computing](../qiu-quantum-computing/index.md) and [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/index.md), and the section-based (`# %%`) scripts in `scripts/`: the simulation, run on a SLURM cluster by the jobs in `scripts/cluster/`, and the analyses producing the figures of the paper.

## Installation

The app is a member of the uv workspace and is not published. It is installed with all other packages, from the repository root:

```sh
uv sync --all-packages
```

Its scripts are run with the workspace environment, e.g. `uv run python apps/wave_optics_propagation/scripts/simulate.py --help`. The analyses typeset their figures with LaTeX, which needs a LaTeX installation.

## Quick start

A small version of the experiment of the paper, 32 transverse samples (5 qubits), 10 lens slices and 3 free space steps, simulated with the Qiskit backend and compared with the exact application of the same phases:

```python
from pathlib import Path

import numpy as np
from qiu_classical_simulation.wave_optics.cli import parse_parameters
from qiu_classical_simulation.wave_optics.simulation import ExactBackend, simulate
from wave_optics_propagation.backend import QiskitBackend

parameters, _ = parse_parameters(
    "Simulate the lens experiment with Qiskit.",
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
result = simulate(parameters, QiskitBackend())
exact = simulate(parameters, ExactBackend())

fidelity = abs(np.vdot(result.snapshots["final"], exact.snapshots["final"])) ** 2
assert fidelity > 0.999
assert 0 < result.total_probability_of_success < 1
```

The protocol reproduces the exact field with a fidelity above 0.999, and all its cycles succeed with a probability of about 0.48.

## Where next

- The [User Guide](user-guide.md) describes the experiment, the Qiskit backend, the scripts, the runs on the cluster, the stored results and the changes to the former analyses.
- The [Examples](examples.md) simulate small experiments, compare them with the classical numerics, store and load them, and compute the beam waist along the propagation.
- The [API Reference](reference/wave_optics_propagation/index.md) documents the backend.
