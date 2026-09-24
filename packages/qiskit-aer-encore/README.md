# Qiskit Aer Encore

Aer simulators configured for the available hardware, used by the tests and applications of this monorepo.

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

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-aer-encore
```

The GPU paths are tested with a faked device list, so the tests also run without a GPU.
