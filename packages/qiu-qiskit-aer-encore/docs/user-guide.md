# User Guide

The package consists of the module [`simulator`][qiu_qiskit_aer_encore.simulator]: the enum [`AerDevice`][qiu_qiskit_aer_encore.simulator.AerDevice] of the devices, the detection of the available devices by [`available_aer_devices`][qiu_qiskit_aer_encore.simulator.available_aer_devices], the default options of GPU simulators [`GPU_OPTIONS`][qiu_qiskit_aer_encore.simulator.GPU_OPTIONS], and the factory [`aer_simulator`][qiu_qiskit_aer_encore.simulator.aer_simulator] combining them.

## Devices

An [`AerDevice`][qiu_qiskit_aer_encore.simulator.AerDevice] selects where a simulator runs:

| device           | raw value | simulates on                                        |
| ---------------- | --------- | --------------------------------------------------- |
| `AUTO` (default) | `"auto"`  | a GPU if one is available, and the CPU otherwise    |
| `CPU`            | `"cpu"`   | the CPU, which is always available                  |
| `GPU`            | `"gpu"`   | a GPU, raising an `AerError` if none is available   |

`AerDevice` is an `ExtendedEnum` of [qiu-python-encore](../../qiu-python-encore/): its members compare equal to their raw values, and [`aer_simulator`][qiu_qiskit_aer_encore.simulator.aer_simulator] accepts either. The raw values are lowercase; other strings, including Aer's own uppercase device names such as `"CPU"`, raise a `ValueError`.

```python
from qiu_qiskit_aer_encore.simulator import AerDevice, aer_simulator

assert AerDevice("cpu") is AerDevice.CPU
assert AerDevice.GPU == "gpu"
assert aer_simulator("cpu").options.device == "CPU"
```

Which device to choose:

- `AUTO` for code that should use the fastest hardware available, e.g. applications run both locally and on a cluster.
- `CPU` for reproducible tests and small circuits, where transferring the state to a GPU does not pay off, or to keep the GPU free.
- `GPU` where a CPU simulation would be too slow to be useful, so that a missing GPU fails early rather than silently falling back to the CPU.

## Detecting the available devices

[`available_aer_devices()`][qiu_qiskit_aer_encore.simulator.available_aer_devices] returns the devices Aer can simulate on, as Aer names them, e.g. `("CPU",)` or `("CPU", "GPU")`. It asks a default `AerSimulator` once, and caches the answer for the lifetime of the process: every later call returns the same tuple. `aer_simulator` relies on it, so the device selection costs nothing after the first call.

```python
from qiu_qiskit_aer_encore.simulator import available_aer_devices

devices = available_aer_devices()
assert "CPU" in devices
assert available_aer_devices() is devices  # detected once
```

A GPU is available only with a GPU build of Aer, such as the `qiskit-aer-gpu` distribution, and a working Nvidia driver. With the plain `qiskit-aer` distribution, the only device is the CPU, and `AUTO` always selects it.

## Creating simulators

[`aer_simulator(device, **options)`][qiu_qiskit_aer_encore.simulator.aer_simulator] returns an `AerSimulator` with its `device` option set to `"CPU"` or `"GPU"`:

1. The device is converted to an `AerDevice`, so raw values are accepted.
2. If the `GPU` is requested but not available, an `AerError` is raised.
3. For `GPU`, or for `AUTO` with a GPU available, the simulator runs on the GPU with the options `GPU_OPTIONS`, updated by the given options.
4. Otherwise, it runs on the CPU with only the given options.

The keyword options are the options of `AerSimulator`, e.g. `method="statevector"`, `precision="single"` or `seed_simulator=1`; they are passed on unchanged, and validated only by Aer, which raises an `AerError` for unknown options. The `device` itself cannot be passed as an option, since it is set by the factory.

```python
from qiu_qiskit_aer_encore.simulator import aer_simulator

simulator = aer_simulator("cpu", method="statevector", precision="single")
assert simulator.options.method == "statevector"
assert simulator.options.precision == "single"
```

The factory creates a new simulator on every call; nothing but the device detection is cached. The returned simulator is an ordinary `AerSimulator`, so everything Aer offers, e.g. `simulator.set_options(...)` or noise models, works as usual.

## Options of GPU simulators

Simulators on GPUs start from the options of [`GPU_OPTIONS`][qiu_qiskit_aer_encore.simulator.GPU_OPTIONS]:

| option               | value  | effect                                                                                      |
| -------------------- | ------ | ------------------------------------------------------------------------------------------- |
| `cuStateVec_enable`  | `True` | accelerates the simulation with Nvidia's cuStateVec library of cuQuantum, if Aer is built with it |
| `blocking_enable`    | `True` | distributes the simulation over all available GPUs (and MPI processes)                     |
| `batched_shots_gpu`  | `True` | distributes the shots over the available GPUs, e.g. for noisy or measured circuits          |

Options passed to `aer_simulator` override these defaults for that simulator only, e.g. `aer_simulator("gpu", blocking_enable=False)` to simulate on a single GPU. The dictionary itself is shared by all simulators created afterwards, so change it only deliberately, e.g. at the start of an application.

!!! note "Options Aer restricts to some methods"
    Aer documents `blocking_enable` only for the `"statevector"`, `"density_matrix"` and `"unitary"` methods, and recommends setting the chunk size `blocking_qubits` along with it; the defaults leave it to Aer. Pass `blocking_qubits=...` where a simulation needs it.

CPU simulators do not use `GPU_OPTIONS` at all: they get exactly the given options.

## Running circuits

The simulators run circuits of the instructions Aer supports, so transpile circuits for the simulator before running them, especially circuits with high-level gates, such as the `StatePreparation` and `QFTGate` of the `GATE` method of [qiu-quantum-computing](../../qiu-quantum-computing/):

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator

circuit = QuantumCircuit(1)
circuit.h(0)
circuit.save_statevector()

simulator = aer_simulator("cpu", method="statevector")
result = simulator.run(transpile(circuit, simulator)).result()
assert np.allclose(result.get_statevector().data, [2**-0.5, 2**-0.5])
```

Aer's save instructions, e.g. `save_statevector`, are added to `QuantumCircuit` by importing `qiskit_aer`, which importing this package does.

### Exact statevectors of transpiled circuits

From optimization level 2, the default of `transpile`, the transpiler may change a circuit in ways that measurements account for, but saved statevectors do not:

- it elides swap gates, e.g. the final swaps of a QFT, into a relabeling of the qubits, the `final_layout` of the transpiled circuit, and `save_statevector` then saves the state with its qubits permuted;
- it removes gates it deems equivalent to the identity, e.g. rotations by angles of `1e-7`, which changes the amplitudes by as much.

Transpile with `optimization_level=1` where the statevector must be exact, as the tests of the monorepo do with `transpile_exactly` of [qiskit-pytest-helper](../../qiskit-pytest-helper/):

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator

circuit = QuantumCircuit(2)
circuit.x(0)
circuit.swap(0, 1)
reference = Statevector(circuit)  # |10>
circuit.save_statevector()

simulator = aer_simulator("cpu", method="statevector")
transpiled = transpile(circuit, simulator, optimization_level=1)
result = simulator.run(transpiled).result()
assert np.allclose(result.get_statevector().data, reference.data)
```

Sampled counts are not affected: the measurements are remapped along with the qubits.

## Pitfalls

- `AUTO` runs on different hardware on different machines. Results of sampling agree statistically, but floating-point details and performance differ; pin the device for reproducible benchmarks.
- `AUTO` selects the GPU regardless of the simulation method, but Aer supports only some methods on GPUs, e.g. `statevector`, `density_matrix` and `unitary`, and not `stabilizer`, `extended_stabilizer`, `matrix_product_state` or `superop`. Pin `device="cpu"` for those methods.
- `GPU` raises an `AerError` without a GPU, rather than falling back to the CPU; catch it where a fallback is wanted.
- The default optimization level of `transpile` may permute the qubits of saved statevectors and drop tiny rotations; use `optimization_level=1` for exact statevectors.
- Device names are the lowercase raw values of `AerDevice`, not Aer's uppercase names.
- The available devices are detected once per process: a GPU that becomes available later is not noticed.
- The options are passed on to Aer without validation by this package; Aer rejects unknown options, e.g. a misspelled one, with an `AerError`.
