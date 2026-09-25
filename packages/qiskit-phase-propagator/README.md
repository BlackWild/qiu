# Qiskit Phase Propagator

[![PyPI](https://img.shields.io/pypi/v/qiskit-phase-propagator)](https://pypi.org/project/qiskit-phase-propagator/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiskit-phase-propagator/)

Circuits applying the phase `e^(i f(x))` of a signal `f` to the basis states of a qubit register, where the signal is a [`python-signals`](https://github.com/BlackWild/qiu/blob/master/shareable-packages/python-signals/README.md) signal on an axis of `2**n` samples.

## Installation

```sh
pip install qiskit-phase-propagator
```

## Encoding of the axis in qubits

`qubit_encoding` fixes how an axis is represented: its `2**n` samples are the basis states `|k>` of `n` qubits (`num_qubits_of`), and `|k>` encodes the integer index `axis.index[k]`, whose bit weights depend on the index ordering of the axis (`bit_weights`):

| ordering   | encoded integer of the bits `x_(n-1) ... x_0`            |
| ---------- | -------------------------------------------------------- |
| `NATURAL`  | unsigned, `sum_i 2^i x_i`                                |
| `FFT`      | two's complement                                         |
| `CENTERED` | two's complement of the bits with the top one flipped    |

## Direct phases

`direct.polynomial_phase_circuit(signal)` applies `e^(i alpha x^power)` exactly for a `PolynomialSignal` of power up to 3, with (multi-)controlled phase gates from expanding the power of the encoded integer (`Order1DirectPhase`, `Order2DirectPhase`, `Order3DirectPhase`).

## Sample-based phases

`sample_based` applies `e^(i f(x))` for an arbitrary real signal of one sign, sampled or algebraic:

1. `sample_based_decomposition(signal)` splits it as `f = alpha |phi|^2`, with `alpha` the sum of its samples.
2. `slice_alpha_to_deltas_evenly(alpha, max_delta)` slices `alpha` into equal small phases `delta`.
3. Each cycle prepares `|phi>` in a second register, applies `e^(i delta)` where both registers agree (`partial_phase_circuit`), un-prepares `|phi>` and measures. On success (`|0...0>`), the amplitudes become `psi_j (1 + (e^(i delta) - 1) |phi_j|^2)`, i.e. `e^(i delta |phi_j|^2) psi_j` up to `O(delta^2)`.

`QuadraticSignalSampleBasedPhasePropagator(signal, max_delta, method)` combines these, with `|phi>` prepared by `qiskit-encore` using the given `SynthesisMethod`. The lower-level `GenericIterativeSampleBasedPhasePropagator` (one cycle per delta, each conditioned on the previous successes) and `GenericIterativeSampleBasedPhasePropagatorWithConstantDelta` (a loop that breaks at the first failure) take the preparation circuits directly.

The `method` defaults to `GATE`, a Qiskit `StatePreparation` synthesized when transpiling. Qiskit's synthesis is unreliable for the nearly uniform `|phi>` of smooth signals (qiskit 2.2): transpiling or simulating it can fail, e.g. in its two-qubit decompositions or its uniformly controlled gates, and it can prepare wrong states (see `qiskit-encore`). Pass `DECOMPOSED` for the Möttönen synthesis of `qiskit-encore`, or `DENSE` for exact results on few qubits.

`sample_based_manual` simulates the same protocol on statevectors, post-selected on success, applying the partial phase as its diagonal (`partial_phase_diagonal`), which is much faster than simulating its multi-controlled phase gate, e.g. `phase_propagate_state_with_arbitrary_signal(psi, signal, max_delta)`; `phase_propagation_cycle` also returns the success probability of a cycle.

## Usage

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_phase_propagator.direct import polynomial_phase_circuit
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)
from qiskit_phase_propagator.sample_based_manual import (
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

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiskit-phase-propagator/>. To serve it locally, from the repository root:

```sh
uv run --group docs mkdocs serve -f packages/qiskit-phase-propagator/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-phase-propagator
```

The circuit tests run on the Aer simulator and compare the exact amplitudes with the closed form of the cycles above.
