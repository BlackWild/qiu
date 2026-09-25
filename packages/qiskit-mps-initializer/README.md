# Qiskit MPS Initializer

[![PyPI](https://img.shields.io/pypi/v/qiskit-mps-initializer)](https://pypi.org/project/qiskit-mps-initializer/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiskit-mps-initializer/)

State preparation with layers of one- and two-qubit gates from matrix product states (MPS), for circuits much shallower than exact state preparation at the cost of an approximation. Also published on PyPI as `qiskit-mps-initializer`.

## Installation

```sh
pip install qiskit-mps-initializer
```

## Layers of MPS gates

`mps` builds the layers (Ran, Phys. Rev. A 101, 032310, 2020):

- `bond2_mps_approximation(state)` truncates a state of 2 qubits or more to an MPS of bond dimension at most 2.
- `disentangler_matrices(mps)` completes the tensors of such an MPS to unitaries, a two-qubit one per site but the last and a single-qubit one for the last.
- `mps_layer(mps)` places them into the circuit `U` with `U|0...0>` the state of the MPS: the two-qubit gates on neighboring qubits from the most significant one down, then the single-qubit gate on qubit 0.

## State preparation

`state_preparation.mps_state_preparation(state, max_layers, tolerance=None)` prepares a state with at most `max_layers` layers. Each layer prepares the bond-2 approximation of what the layers before it leave, the residual `U_k^dagger ... U_1^dagger |state>`, until the residual reaches `|0...0>`: by default when it equals it as a Qiskit `Statevector`, up to Qiskit's tolerances, or within the distance `tolerance`. The result holds the `circuit`, its `layers`, the `error` of the prepared state, i.e. its distance to the target including the global phase, and whether it `converged`.

States of up to 3 qubits and product states are prepared exactly by one layer. For more qubits, the error decreases with the number of layers, though not always monotonically, and slowly for highly entangled states.

## Usage

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_mps_initializer.state_preparation import mps_state_preparation

rng = np.random.default_rng(0)
state = rng.normal(size=32) + 1j * rng.normal(size=32)
state /= np.linalg.norm(state)

preparation = mps_state_preparation(state, max_layers=20)
fidelity = abs(np.vdot(Statevector(preparation.circuit).data, state)) ** 2
assert fidelity > 0.99 and preparation.num_layers == 20
```

The state of an intensity signal `f = alpha |psi|**2`, e.g. of a `python-signals` signal, follows from `qiskit_phase_propagator.sample_based.sample_based_decomposition(signal)`.

## Changes from version 0.2

The functionality of version 0.2 is carried over onto the packages of this monorepo:

| 0.2                                                     | now                                                                   |
| ------------------------------------------------------- | --------------------------------------------------------------------- |
| `QuantumState.from_dense_data(data, normalize)`         | a Qiskit `Statevector`, validated by `qiskit_encore`                  |
| `QuantumState.generate_mps_initializer_circuit(layers)` | `mps_state_preparation(state, max_layers).circuit`                    |
| `QuantumIntensity`                                      | `qiskit_phase_propagator.sample_based.sample_based_decomposition`     |
| `helpers.mps_technique.G_matrices`                      | `mps.disentangler_matrices`                                           |
| `helpers.mps_technique.multi_layered_circuit_for_non_approximated` | `mps_state_preparation`                                    |
| `helpers.sampling_and_data_preperation`                 | `python_signals` axes and `AlgebraicSignal.from_sympy`                |
| `utils.simulate_statevector`, `simulate_quantum_info`   | `Statevector(circuit)`, or `qiskit_aer_encore.simulator.aer_simulator` |

Besides:

- `max_layers` is required and bounds the number of layers; formerly, one layer more than asked for could be added, and without a maximum the construction could run forever.
- Single-qubit states are supported.
- The package no longer depends on `pydantic`, `pydantic-numpy` and `qiskit-aer`.

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiskit-mps-initializer/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiskit-mps-initializer/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-mps-initializer
```
