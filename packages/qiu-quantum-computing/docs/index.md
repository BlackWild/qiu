# qiu-quantum-computing

`qiu-quantum-computing` provides generic computations with the amplitudes of a quantum computer, as Qiskit circuits, without reference to the physical dynamics they may simulate:

- **Circuit building blocks**: state preparation from the all-zero state, the quantum Fourier transform (QFT) and uniformly controlled rotations. The state preparation and the QFT can be represented in their circuit in three ways, chosen by the [`SynthesisMethod`](../qiu-qiskit-encore/) of qiu-qiskit-encore: as a dense unitary, as a high-level Qiskit gate, or as a circuit of elementary gates synthesized by this package. Its own synthesis of state preparation works around a bug in Qiskit's `StatePreparation`, see the [User Guide](user-guide.md#gate-qiskits-statepreparation).
- **Diagonal phase operators**: the subpackage `phase_propagator` applies the phase `e^(i f(x))` of a [qiu-signals](../qiu-signals/) signal `f` on an axis of `2**n` samples to the basis states of `n` qubits. The direct phases apply `e^(i alpha x^power)` exactly for monomials up to the power 3, with (multi-)controlled phase gates. The sample-based phases apply `e^(i f)` for an arbitrary real signal of one sign, sampled or algebraic, by repeating a probabilistic cycle that prepares `|phi> = sqrt(f / alpha)` in a second register; a statevector simulation of the same protocol, post-selected on success, is provided for fast numerics.

In the monorepo, it is the base of the simulation of physical systems: [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/) evolves states under potentials and kinetic energies with its phase operators and QFT, [qiu-mps-initializer](../qiu-mps-initializer/) complements its exact state preparation, and the Qiskit backend of the lens experiments in [wave_optics_propagation](../wave_optics_propagation/) is built on its phase operators.

## Installation

```sh
pip install qiu-quantum-computing
```

Inside the monorepo, it is installed with the other workspace packages by `uv sync --all-packages`.

## Quick start

A state prepared exactly, and transformed by the QFT:

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.qft import qft_circuit
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

state = np.array([0.5, 0.5j, -0.5, -0.5j])

# a circuit of ry, rz and cx gates preparing the state exactly, global phase included
circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
assert np.allclose(Statevector(circuit).data, state)

# the QFT acts on the amplitudes as NumPy's orthonormal inverse DFT
circuit.compose(qft_circuit(2), inplace=True)
assert np.allclose(Statevector(circuit).data, np.fft.ifft(state, norm="ortho"))
```

A thin lens multiplies a field by a quadratic phase. On 3 qubits, the direct phase circuit applies it exactly:

```python
import numpy as np
from qiu_signals.algebraic_signal import QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.phase_propagator.direct import polynomial_phase_circuit

x_axis = PositionAxis(size=8, delta_x=0.25, ordering=IndexOrdering.CENTERED)
lens = QuadraticSignal(x_axis, alpha=-2.0)

psi = Statevector(np.ones(8) / np.sqrt(8))
out = psi.evolve(polynomial_phase_circuit(lens))

assert np.allclose(out.data, np.exp(1j * lens.data) * psi.data)
```

## Where next

- The [User Guide](user-guide.md) explains the conventions and the pitfalls of each building block, and for the phase propagator the qubit encoding of the axes, the direct phases, the sample-based protocol with its closed-form cycle map and success probability, and how to run the circuits.
- The [Examples](examples.md) show worked use cases, from comparing the synthesis methods and measuring the overlap of two states to a shifted lens and the propagators on the Aer simulator.
- The [API Reference](reference/qiu_quantum_computing/index.md) documents every module, class and function, including those of the [`phase_propagator`](reference/qiu_quantum_computing/phase_propagator/index.md).
