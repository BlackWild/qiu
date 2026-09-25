# Qiu Quantum Computing

[![PyPI](https://img.shields.io/pypi/v/qiu-quantum-computing)](https://pypi.org/project/qiu-quantum-computing/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiu-quantum-computing/)

Generic computations with the amplitudes of a quantum computer, as Qiskit circuits, without reference to the physical dynamics they may simulate: state preparation, the quantum Fourier transform (QFT), uniformly controlled rotations, and diagonal phase operators `e^(i f(x))` of sampled signals. The state preparation and the QFT are represented as chosen by the `SynthesisMethod` of [`qiu-qiskit-encore`](https://github.com/BlackWild/qiu/blob/master/packages/qiu-qiskit-encore/README.md): as a dense unitary, a Qiskit gate or a decomposed circuit.

## Installation

```sh
pip install qiu-quantum-computing
```

## State preparation

`state_preparation.state_preparation_circuit(state, *, method=..., inverse=False)` returns a circuit `U` with `U|0...0> = |state>` exactly, global phase included, or with `inverse=True` the circuit mapping the state back to `|0...0>`.

- `GATE` (default) appends Qiskit's `StatePreparation`. **Caveat:** Qiskit's isometry synthesis prepares wrong states (fidelity 0) when two of its intermediate single-qubit gates are close but not equal. A test documents this with a state found by `hypothesis`, and alerts once Qiskit fixes it. Use `DECOMPOSED` where this matters.
- `DECOMPOSED` uses `decomposed_state_preparation`, the synthesis of Möttönen et al. (2005) with uniformly controlled `ry` and `rz` rotations: at most `2**(n+1) - 4` CNOT gates, and no `rz` gates at all for real non-negative states. All angles come from `arctan2` and sums, so it is numerically robust for any state.
- `DENSE` appends the unitary of `state_preparation_unitary`, a Householder reflection with the state as its first column: exact, stable and `O(4**n)`.

`preparable_state.PreparableState(statevector, method)` bundles a state with its method, and caches its preparation `circuit` and `inverse_circuit`. It is immutable, so the cached circuits always prepare the state.

## Quantum Fourier transform

`qft.qft_circuit(num_qubits, inverse=False, method=...)` returns the QFT, with `GATE` (default) being a `QFTGate`, `DECOMPOSED` Qiskit's `synth_qft_full` (Hadamard, controlled phase and swap gates) and `DENSE` the matrix of `qft_matrix`.

The QFT maps `|j>` to `sum_k e^(2 pi i j k / N) |k> / sqrt(N)`, with the integers encoded in little-endian order like the statevector indices. On the amplitudes, it is thus NumPy's orthonormal **inverse** DFT, `numpy.fft.ifft(psi, norm="ortho")`, and the inverse QFT is `numpy.fft.fft(psi, norm="ortho")`.

## Uniformly controlled rotations

`uniformly_controlled_rotation.uniformly_controlled_rotation(axis, angles)` returns the multiplexed `ry` or `rz` rotation, `diag(R(angles[0]), R(angles[1]), ...)`, on the target qubit 0 controlled by the qubits `1, ..., k`. It uses `2**k` rotations and `2**k` CNOT gates (Möttönen et al., 2004), and no CNOT gates at all if the angles do not depend on the controls.

## Phase propagator

The subpackage `phase_propagator` applies the phase `e^(i f(x))` of a signal `f` to the basis states of a qubit register, where the signal is a [`qiu-signals`](https://github.com/BlackWild/qiu/blob/master/packages/qiu-signals/README.md) signal on an axis of `2**n` samples.

### Encoding of the axis in qubits

`qubit_encoding` fixes how an axis is represented: its `2**n` samples are the basis states `|k>` of `n` qubits (`num_qubits_of`), and `|k>` encodes the integer index `axis.index[k]`, whose bit weights depend on the index ordering of the axis (`bit_weights`):

| ordering   | encoded integer of the bits `x_(n-1) ... x_0`            |
| ---------- | -------------------------------------------------------- |
| `NATURAL`  | unsigned, `sum_i 2^i x_i`                                |
| `FFT`      | two's complement                                         |
| `CENTERED` | two's complement of the bits with the top one flipped    |

### Direct phases

`direct.polynomial_phase_circuit(signal)` applies `e^(i alpha x^power)` exactly for a `PolynomialSignal` of power up to 3, with (multi-)controlled phase gates from expanding the power of the encoded integer (`Order1DirectPhase`, `Order2DirectPhase`, `Order3DirectPhase`).

### Sample-based phases

`sample_based` applies `e^(i f(x))` for an arbitrary real signal of one sign, sampled or algebraic:

1. `sample_based_decomposition(signal)` splits it as `f = alpha |phi|^2`, with `alpha` the sum of its samples.
2. `slice_alpha_to_deltas_evenly(alpha, max_delta)` slices `alpha` into equal small phases `delta`.
3. Each cycle prepares `|phi>` in a second register, applies `e^(i delta)` where both registers agree (`partial_phase_circuit`), un-prepares `|phi>` and measures. On success (`|0...0>`), the amplitudes become `psi_j (1 + (e^(i delta) - 1) |phi_j|^2)`, i.e. `e^(i delta |phi_j|^2) psi_j` up to `O(delta^2)`.

`QuadraticSignalSampleBasedPhasePropagator(signal, max_delta, method)` combines these, with `|phi>` prepared by a `PreparableState` of the given `SynthesisMethod`. The lower-level `GenericIterativeSampleBasedPhasePropagator` (one cycle per delta, each conditioned on the previous successes) and `GenericIterativeSampleBasedPhasePropagatorWithConstantDelta` (a loop that breaks at the first failure) take the preparation circuits directly.

The `method` defaults to `GATE`, a Qiskit `StatePreparation` synthesized when transpiling. Qiskit's synthesis is unreliable for the nearly uniform `|phi>` of smooth signals (qiskit 2.2): transpiling or simulating it can fail, e.g. in its two-qubit decompositions or its uniformly controlled gates, and it can prepare wrong states (see the state preparation above). Pass `DECOMPOSED` for the Möttönen synthesis, or `DENSE` for exact results on few qubits.

`sample_based_manual` simulates the same protocol on statevectors, post-selected on success, applying the partial phase as its diagonal (`partial_phase_diagonal`), which is much faster than simulating its multi-controlled phase gate, e.g. `phase_propagate_state_with_arbitrary_signal(psi, signal, max_delta)`; `phase_propagation_cycle` also returns the success probability of a cycle.

## Usage

The building blocks:

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_quantum_computing.qft import qft_circuit
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

state = np.array([0.5, 0.5j, -0.5, -0.5j])

# a gate description, synthesized by Qiskit later on
circuit = state_preparation_circuit(state)
assert np.allclose(Statevector(circuit).data, state)

# elementary ry, rz and cx gates of our own, robust synthesis
decomposed = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)

# a state bundled with its (cached) preparation and un-preparation circuits
phi = PreparableState(state, method=SynthesisMethod.DENSE)
assert np.allclose(Statevector(state).evolve(phi.inverse_circuit).data, [1, 0, 0, 0])

# the QFT is NumPy's orthonormal inverse DFT
assert np.allclose(
    Statevector(state).evolve(qft_circuit(2)).data, np.fft.ifft(state, norm="ortho")
)
```

The phase propagator:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.phase_propagator.direct import polynomial_phase_circuit
from qiu_quantum_computing.phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)
from qiu_quantum_computing.phase_propagator.sample_based_manual import (
    phase_propagate_state_with_arbitrary_signal,
)

x_axis = PositionAxis(size=8, delta_x=0.25, ordering=IndexOrdering.CENTERED)
psi = Statevector(np.ones(8) / np.sqrt(8))

# an exact quadratic phase
lens = QuadraticSignal(x_axis, alpha=0.5)
out = psi.evolve(polynomial_phase_circuit(lens))
assert np.allclose(out.data, np.exp(1j * lens.data) * psi.data)

# an arbitrary positive phase, applied sample-based
potential = AlgebraicSignal(x_axis, lambda x: 0.02 * (1 + np.cos(x)))
propagator = QuadraticSignalSampleBasedPhasePropagator(potential, max_delta=0.01)
simulated = phase_propagate_state_with_arbitrary_signal(psi, potential, max_delta=0.01)
assert abs(np.vdot(simulated.data, np.exp(1j * potential.data) * psi.data)) > 0.9999
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiu-quantum-computing/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiu-quantum-computing/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiu-quantum-computing
```

The circuit tests run on the Aer simulator; those of the phase propagator compare the exact amplitudes with the closed form of its cycles.
