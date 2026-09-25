# Qiskit Aer Encore

[![PyPI](https://img.shields.io/pypi/v/qiskit-aer-encore)](https://pypi.org/project/qiskit-aer-encore/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiskit-aer-encore/)

Aer simulators configured for the available hardware, used by the tests of this monorepo.

## Installation

```sh
pip install qiskit-aer-encore
```

## Usage

`simulator.aer_simulator(device=..., **options)` returns an `AerSimulator` on the requested `AerDevice`:

| device           | simulates on                                             |
| ---------------- | -------------------------------------------------------- |
| `AUTO` (default) | a GPU if one is available, and the CPU otherwise         |
| `CPU`            | the CPU, which is always available                       |
| `GPU`            | a GPU, raising an `AerError` if none is available        |

On GPUs, the simulator uses Nvidia's cuStateVec library and distributes the simulation and the shots over all available devices (`GPU_OPTIONS`). Further keyword options are passed on to `AerSimulator`, overriding these defaults. The available devices are detected once, see `available_aer_devices()`.

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer_encore.simulator import aer_simulator

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

simulator = aer_simulator(device="cpu", method="statevector")
counts = simulator.run(transpile(circuit, simulator), shots=100).result().get_counts()
assert set(counts) <= {"00", "11"}
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiskit-aer-encore/>. To serve it locally, from the repository root:

```sh
uv run --group docs mkdocs serve -f packages/qiskit-aer-encore/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-aer-encore
```

The GPU paths are tested with a faked device list, so the tests also run without a GPU.
