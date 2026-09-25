# qiskit-mps-initializer

`qiskit-mps-initializer` prepares quantum states approximately with layers of one- and two-qubit gates obtained from matrix product states (MPS), following Ran, "Encoding of matrix product states into quantum circuits of one- and two-qubit gates", Phys. Rev. A 101, 032310 (2020). Each layer acts on neighboring qubits only and costs at most `3 (n - 1)` CNOT gates on `n` qubits, so a few layers give circuits much shallower than exact state preparation, whose CNOT count grows as `2**n`, at the cost of an approximation. Every preparation reports its error.

Within the monorepo, it builds on [qiskit-encore](../qiskit-encore/), which validates its states and prepares single-qubit states exactly, and complements the exact state preparation there. The states of intensity signals of [python-signals](../python-signals/) come from [qiskit-phase-propagator](../qiskit-phase-propagator/). Version 0.3 carries the functionality of version 0.2 over onto these packages, see the [User Guide](user-guide.md#changes-from-version-02).

## Installation

```sh
pip install qiskit-mps-initializer
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_mps_initializer.state_preparation import mps_state_preparation

rng = np.random.default_rng(0)
state = rng.normal(size=32) + 1j * rng.normal(size=32)
state /= np.linalg.norm(state)

preparation = mps_state_preparation(state, max_layers=8)
prepared = Statevector(preparation.circuit).data
assert preparation.num_layers == 8
assert np.isclose(preparation.error, np.linalg.norm(prepared - state))
assert abs(np.vdot(state, prepared)) ** 2 > 0.99  # the fidelity
```

## Where next

- The [User Guide](user-guide.md) explains the MPS layers, the multi-layer preparation, its error and its convergence.
- The [Examples](examples.md) show worked use cases, from the fidelity of increasing numbers of layers to preparing the state of an intensity signal.
- The [API Reference](reference/qiskit_mps_initializer/index.md) documents every module, class and function.
