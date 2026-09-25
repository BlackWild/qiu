# Examples

## Sampling a GHZ state on the CPU

Sampling measurement outcomes is the most common use of a simulator. This example prepares a GHZ state of 5 qubits, samples it on the CPU with a fixed seed, and checks that only the outcomes `00000` and `11111` occur, about equally often.

```python
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator

num_qubits = 5
circuit = QuantumCircuit(num_qubits)
circuit.h(0)
for qubit in range(num_qubits - 1):
    circuit.cx(qubit, qubit + 1)
circuit.measure_all()

shots = 4000
simulator = aer_simulator(device="cpu", seed_simulator=42)
counts = simulator.run(transpile(circuit, simulator), shots=shots).result().get_counts()

assert set(counts) == {"0" * num_qubits, "1" * num_qubits}
assert abs(counts["0" * num_qubits] / shots - 0.5) < 0.05
```

Both outcomes occur with frequencies close to `1/2`, and the seed makes the counts reproducible from run to run.

## Hardware-independent code with `AUTO`

Code that should run on a laptop as well as on a GPU node leaves the device to `AUTO`. This example creates the simulator without a device, reports where it runs, and computes an exact statevector with Aer's `save_statevector`, which is the same on either device up to floating-point rounding.

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator, available_aer_devices

circuit = QuantumCircuit(3)
circuit.h(range(3))
circuit.cz(0, 2)
circuit.rx(0.3, 1)
reference = Statevector(circuit)
circuit.save_statevector()

simulator = aer_simulator(method="statevector")  # the device defaults to AUTO
expected_device = "GPU" if "GPU" in available_aer_devices() else "CPU"
assert simulator.options.device == expected_device

statevector = simulator.run(transpile(circuit, simulator)).result().get_statevector()
assert np.allclose(statevector.data, reference.data)
```

The simulator runs on the GPU where one is available and on the CPU otherwise, and its statevector agrees with Qiskit's reference `Statevector(circuit)`, global phase included.

## Falling back when no GPU is available

An application may prefer to fail loudly without a GPU in production, but fall back to the CPU during development. Requesting the `GPU` explicitly raises an `AerError` without one, which the caller can catch.

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerError
from qiu_qiskit_aer_encore.simulator import (
    AerDevice,
    aer_simulator,
    available_aer_devices,
)


def simulator_for(require_gpu: bool):
    """Return a GPU simulator, or a CPU one unless a GPU is required."""
    try:
        return aer_simulator(AerDevice.GPU)
    except AerError:
        if require_gpu:
            raise
        return aer_simulator(AerDevice.CPU)


simulator = simulator_for(require_gpu=False)
if "GPU" not in available_aer_devices():
    assert simulator.options.device == "CPU"

circuit = QuantumCircuit(1)
circuit.x(0)
circuit.measure_all()
counts = simulator.run(transpile(circuit, simulator), shots=10).result().get_counts()
assert counts == {"1": 10}
```

Without a GPU, `simulator_for(require_gpu=False)` returns a CPU simulator, while `simulator_for(require_gpu=True)` would raise the `AerError`.

## Many qubits with the matrix product state method

The keyword options select any of Aer's simulation methods. A statevector of 40 qubits would need terabytes of memory, but the GHZ state has little entanglement, so Aer's `matrix_product_state` method samples it in a fraction of a second.

```python
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator

num_qubits = 40
circuit = QuantumCircuit(num_qubits)
circuit.h(0)
for qubit in range(num_qubits - 1):
    circuit.cx(qubit, qubit + 1)
circuit.measure_all()

simulator = aer_simulator(device="cpu", method="matrix_product_state")
assert simulator.options.method == "matrix_product_state"

result = simulator.run(transpile(circuit, simulator), shots=200, seed_simulator=1)
counts = result.result().get_counts()
assert set(counts) == {"0" * num_qubits, "1" * num_qubits}
```

Only the two GHZ outcomes of 40 bits occur. Aer supports the `matrix_product_state` method only on the CPU, which is why the example pins the device instead of leaving it to `AUTO`.

## Simulating a building block of qiu-quantum-computing

The circuits of [qiu-quantum-computing](../qiu-quantum-computing/index.md) run on these simulators like any other circuit. This example prepares a random state of 4 qubits with the `DECOMPOSED` synthesis, applies the QFT as a high-level `QFTGate`, which the transpiler synthesizes for the simulator, and checks the simulated statevector against NumPy's orthonormal inverse DFT. It transpiles with `optimization_level=1`, which keeps the saved statevector exact (see the [User Guide](user-guide.md#exact-statevectors-of-transpiled-circuits)).

```python
import numpy as np
from qiskit import transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_quantum_computing.qft import qft_circuit
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

rng = np.random.default_rng(5)
state = rng.normal(size=16) + 1j * rng.normal(size=16)
state /= np.linalg.norm(state)

circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
circuit.compose(qft_circuit(4), inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector")
transpiled = transpile(circuit, simulator, optimization_level=1)
result = simulator.run(transpiled).result()
assert np.allclose(result.get_statevector().data, np.fft.ifft(state, norm="ortho"))
```

The simulated amplitudes equal `numpy.fft.ifft(state, norm="ortho")`, since both the state preparation and the QFT of qiu-quantum-computing are exact, global phase included.
