# qiskit-phase-propagator

Circuits applying the phase `e^(i f(x))` of a signal `f` to the basis states of a qubit register. The signal is a [`python-signals`](../python-signals/) signal on an axis of `2**n` samples, whose sample `k` is the basis state `|k>` of `n` qubits.

Two kinds of circuits are provided. The direct phases apply `e^(i alpha x^power)` exactly for monomials up to the power 3, with (multi-)controlled phase gates. The sample-based phases apply `e^(i f)` for an arbitrary real signal of one sign, sampled or algebraic, by repeating a probabilistic cycle that prepares `|phi> = sqrt(f / alpha)` in a second register; a statevector simulation of the same protocol, post-selected on success, is provided for fast numerics.

In the monorepo, the phase circuits are the building blocks of [`qiskit-hamiltonian-simulation`](../qiskit-hamiltonian-simulation/), which evolves states under potentials and kinetic energies, and of the Qiskit backend of the lens experiments in [`wave_optics_propagation`](../wave_optics_propagation/). The state preparations and synthesis methods come from [`qiskit-encore`](../qiskit-encore/).

## Installation

```sh
pip install qiskit-phase-propagator
```

Inside the monorepo, it is installed with the other workspace packages by `uv sync --all-packages`.

## Quick start

A thin lens multiplies a field by a quadratic phase. On 3 qubits, the direct phase circuit applies it exactly:

```python
import numpy as np
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_phase_propagator.direct import polynomial_phase_circuit

x_axis = PositionAxis(size=8, delta_x=0.25, ordering=IndexOrdering.CENTERED)
lens = QuadraticSignal(x_axis, alpha=-2.0)

psi = Statevector(np.ones(8) / np.sqrt(8))
out = psi.evolve(polynomial_phase_circuit(lens))

assert np.allclose(out.data, np.exp(1j * lens.data) * psi.data)
```

## Where next

- The [User Guide](user-guide.md) explains the qubit encoding of the axes, the direct phases, the sample-based protocol with its closed-form cycle map and success probability, and how to run the circuits.
- The [Examples](examples.md) work through a shifted lens, an arbitrary phase checked against its closed form, and the propagators on the Aer simulator.
- The [API Reference](reference/qiskit_phase_propagator/index.md) documents every module: [`qubit_encoding`](reference/qiskit_phase_propagator/qubit_encoding.md), [`direct`](reference/qiskit_phase_propagator/direct.md), [`sample_based`](reference/qiskit_phase_propagator/sample_based.md) and [`sample_based_manual`](reference/qiskit_phase_propagator/sample_based_manual.md).
