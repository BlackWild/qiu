# qiu-qiskit-aer-encore

`qiu-qiskit-aer-encore` creates Qiskit Aer simulators configured for the available hardware. A single call, [`aer_simulator`][qiu_qiskit_aer_encore.simulator.aer_simulator], returns an `AerSimulator` on a GPU if one is available and on the CPU otherwise, with options tuned for multi-GPU simulations, so the same code runs on a laptop and on a GPU cluster node.

Within the monorepo, the tests of the quantum packages, e.g. of [qiu-quantum-computing](../qiu-quantum-computing/), [qiu-mps-initializer](../qiu-mps-initializer/) and [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/), run their circuits on the simulators of this package, directly or through the helpers of [qiskit-pytest-helper](../qiskit-pytest-helper/). It depends only on `qiskit-aer` and [qiu-python-encore](../qiu-python-encore/).

## Installation

```sh
pip install qiu-qiskit-aer-encore
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`. Simulating on GPUs additionally needs a GPU build of Aer, e.g. the `qiskit-aer-gpu` distribution in place of `qiskit-aer`, on a machine with an Nvidia GPU.

## Quick start

```python
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

simulator = aer_simulator()  # a GPU if one is available, the CPU otherwise
counts = simulator.run(transpile(circuit, simulator), shots=100).result().get_counts()
assert set(counts) <= {"00", "11"}
```

## Where next

- The [User Guide](user-guide.md) explains the device selection, the default GPU options and how to pass options to the simulator.
- The [Examples](examples.md) show worked use cases, from sampling on the CPU to falling back when no GPU is available.
- The [API Reference](reference/qiu_qiskit_aer_encore/index.md) documents the module and its functions.
