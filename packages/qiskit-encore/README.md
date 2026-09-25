# Qiskit Encore

[![PyPI](https://img.shields.io/pypi/v/qiskit-encore)](https://pypi.org/project/qiskit-encore/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiskit-encore/)

Circuit building blocks on top of Qiskit, used by the other packages of this monorepo. The contributors to this package believe that such tools should be part of the standard Qiskit library, or at least they can envision them being so.

## Installation

```sh
pip install qiskit-encore
```

## Synthesis methods

Every building block can be represented in its circuit in three ways, chosen by a `method` parameter (`synthesis_method.SynthesisMethod`):

| method       | the circuit contains                                                     | use it for                                                         |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `DENSE`      | a single unitary gate defined by its dense matrix                        | exact reference results on few qubits (the matrix has `4**n` entries) |
| `GATE`       | a single high-level Qiskit gate, synthesized by Qiskit when transpiling  | the default: leaving the synthesis to Qiskit, e.g. for hardware-aware transpiling |
| `DECOMPOSED` | an already decomposed circuit of elementary gates                        | a synthesis we control, where Qiskit's is not suitable |

## State preparation

`state_preparation.state_preparation_circuit(state, *, method=..., inverse=False)` returns a circuit `U` with `U|0...0> = |state>` exactly, global phase included, or with `inverse=True` the circuit mapping the state back to `|0...0>`.

- `GATE` (default) appends Qiskit's `StatePreparation`. **Caveat:** Qiskit's isometry synthesis prepares wrong states (fidelity 0) when two of its intermediate single-qubit gates are close but not equal. A test documents this with a state found by `hypothesis`, and alerts once Qiskit fixes it. Use `DECOMPOSED` where this matters.
- `DECOMPOSED` uses `decomposed_state_preparation`, the synthesis of Möttönen et al. (2005) with uniformly controlled `ry` and `rz` rotations: at most `2**(n+1) - 4` CNOT gates, and no `rz` gates at all for real non-negative states. All angles come from `arctan2` and sums, so it is numerically robust for any state.
- `DENSE` appends the unitary of `state_preparation_unitary`, a Householder reflection with the state as its first column: exact, stable and `O(4**n)`.

`preparable_state.PreparableState(statevector, method)` bundles a state with its method, and caches its preparation `circuit` and `inverse_circuit`. It is immutable, so the cached circuits always prepare the state.

## Quantum Fourier transform

`qft.qft_circuit(num_qubits, inverse=False, method=...)` returns the QFT, with `GATE` (default) being a `QFTGate`, `DECOMPOSED` Qiskit's `synth_qft_full` (Hadamard, controlled phase and swap gates) and `DENSE` the matrix of `qft_matrix`.

The QFT maps `|j>` to `sum_k e^(2 pi i j k / N) |k> / sqrt(N)`, with the integers encoded in little-endian order like the statevector indices. On the amplitudes, it is thus NumPy's orthonormal **inverse** DFT, `numpy.fft.ifft(psi, norm="ortho")`, and the inverse QFT is `numpy.fft.fft(psi, norm="ortho")`.

## Uniformly controlled rotations

`uniformly_controlled_rotation.uniformly_controlled_rotation(axis, angles)` returns the multiplexed `ry` or `rz` rotation, `diag(R(angles[0]), R(angles[1]), ...)`, on the target qubit 0 controlled by the qubits `1, ..., k`. It uses `2**k` rotations and `2**k` CNOT gates (Möttönen et al., 2004), and no CNOT gates at all if the angles do not depend on the controls.

## Usage

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.qft import qft_circuit
from qiskit_encore.state_preparation import state_preparation_circuit
from qiskit_encore.synthesis_method import SynthesisMethod

state = np.array([0.5, 0.5j, -0.5, -0.5j])

# a gate description, synthesized by Qiskit later on
circuit = state_preparation_circuit(state)
assert np.allclose(Statevector(circuit).data, state)

# elementary ry, rz and cx gates of our own, robust synthesis
decomposed = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)

# a state bundled with its (cached) preparation and un-preparation circuits
phi = PreparableState(state, method=SynthesisMethod.DENSE)
assert np.allclose(Statevector(state).evolve(phi.inverse_circuit).data, [1, 0, 0, 0])

# the QFT is NumPy's orthonormal inverse DFT
assert np.allclose(
    Statevector(state).evolve(qft_circuit(2)).data, np.fft.ifft(state, norm="ortho")
)
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiskit-encore/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiskit-encore/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-encore
```
