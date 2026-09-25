# qiu-qiskit-encore

`qiu-qiskit-encore` improves native types of Qiskit: [`validated_statevector`][qiu_qiskit_encore.statevector.validated_statevector] returns a Qiskit `Statevector` checked to be a normalized state of qubits, and a [`SynthesisMethod`][qiu_qiskit_encore.synthesis_method.SynthesisMethod] chooses how a circuit building block is represented in a circuit: as a dense unitary, as a high-level Qiskit gate, or as a circuit of elementary gates. The contributors believe that such tools should be part of the standard Qiskit library, or at least can envision them being so.

Within the monorepo, it is the base of the quantum packages: the circuit building blocks of [qiu-quantum-computing](../qiu-quantum-computing/index.md), e.g. its state preparation and QFT, take a `SynthesisMethod`, and they, [qiu-mps-initializer](../qiu-mps-initializer/index.md) and [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/index.md) validate their states with it. It depends only on Qiskit, NumPy and [qiu-python-encore](../qiu-python-encore/index.md).

## Installation

```sh
pip install qiu-qiskit-encore
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

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

## Where next

- The [User Guide](user-guide.md) explains the validation of statevectors and the synthesis methods.
- The [Examples](examples.md) show how to validate the input of a function, and how to write a building block taking a `SynthesisMethod`.
- The [API Reference](reference/qiu_qiskit_encore/index.md) documents every module, class and function.
