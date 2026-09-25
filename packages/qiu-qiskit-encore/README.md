# Qiu Qiskit Encore

[![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-encore)](https://pypi.org/project/qiu-qiskit-encore/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiu-qiskit-encore/)

Improvements of the native types of Qiskit, used by the other packages of this monorepo: statevectors validated as normalized states of qubits, and the synthesis methods choosing how a circuit building block is represented in a circuit. The contributors to this package believe that such tools should be part of the standard Qiskit library, or at least they can envision them being so. The circuit building blocks themselves, e.g. state preparation and the QFT, are in [`qiu-quantum-computing`](https://github.com/BlackWild/qiu/blob/master/packages/qiu-quantum-computing/README.md).

## Installation

```sh
pip install qiu-qiskit-encore
```

## Validated statevectors

`statevector.validated_statevector(state)` returns a copy of a Qiskit `Statevector`, or of amplitudes, as a complex `Statevector`, and raises a `ValueError` unless its dimension is a power of 2 of at least one qubit and it is normalized up to Qiskit's tolerances. States are never normalized silently.

## Synthesis methods

`synthesis_method.SynthesisMethod` chooses how a circuit building block is represented in its circuit, by a `method` parameter:

| method       | the circuit contains                                                     | use it for                                                         |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `DENSE`      | a single unitary gate defined by its dense matrix                        | exact reference results on few qubits (the matrix has `4**n` entries) |
| `GATE`       | a single high-level Qiskit gate, synthesized by Qiskit when transpiling  | the default: leaving the synthesis to Qiskit, e.g. for hardware-aware transpiling |
| `DECOMPOSED` | an already decomposed circuit of elementary gates                        | a synthesis we control, where Qiskit's is not suitable |

It is an `ExtendedEnum` of `qiu-python-encore`, so its members compare equal to their raw values `"dense"`, `"gate"` and `"decomposed"`.

## Usage

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

# a copy of the amplitudes as a statevector, checked to be a normalized state of qubits
statevector = validated_statevector(np.array([0.6, 0.8j]))
assert statevector == Statevector([0.6, 0.8j])

# the synthesis methods compare equal to, and are constructed from, their raw values
assert SynthesisMethod("decomposed") is SynthesisMethod.DECOMPOSED
assert SynthesisMethod.GATE == "gate"
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiu-qiskit-encore/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiu-qiskit-encore/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiu-qiskit-encore
```
