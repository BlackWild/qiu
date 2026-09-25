# qiskit-encore

`qiskit-encore` provides circuit building blocks on top of Qiskit: state preparation from the all-zero state, the quantum Fourier transform (QFT) and uniformly controlled rotations. Each block can be represented in its circuit in three ways, chosen by a [`SynthesisMethod`][qiskit_encore.synthesis_method.SynthesisMethod]: as a dense unitary, as a high-level Qiskit gate, or as a circuit of elementary gates synthesized by this package. The contributors believe that such tools should be part of the standard Qiskit library, or at least can envision them being so.

Within the monorepo, it is the base of the quantum packages: [qiskit-phase-propagator](../qiskit-phase-propagator/) and [qiskit-hamiltonian-simulation](../qiskit-hamiltonian-simulation/) prepare their states with it, and [qiskit-mps-initializer](../qiskit-mps-initializer/) validates its states with it. Its own synthesis of state preparation works around a bug in Qiskit's `StatePreparation`, see the [User Guide](user-guide.md#gate-qiskits-statepreparation).

## Installation

```sh
pip install qiskit-encore
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_encore.qft import qft_circuit
from qiskit_encore.state_preparation import state_preparation_circuit
from qiskit_encore.synthesis_method import SynthesisMethod

state = np.array([0.5, 0.5j, -0.5, -0.5j])

# a circuit of ry, rz and cx gates preparing the state exactly, global phase included
circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
assert np.allclose(Statevector(circuit).data, state)

# the QFT acts on the amplitudes as NumPy's orthonormal inverse DFT
circuit.compose(qft_circuit(2), inplace=True)
assert np.allclose(Statevector(circuit).data, np.fft.ifft(state, norm="ortho"))
```

## Where next

- The [User Guide](user-guide.md) explains the synthesis methods, the conventions and the pitfalls of each building block.
- The [Examples](examples.md) show worked use cases, from comparing the synthesis methods to measuring the overlap of two states.
- The [API Reference](reference/qiskit_encore/index.md) documents every module, class and function.
