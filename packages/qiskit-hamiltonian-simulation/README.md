# Qiskit Hamiltonian Simulation

[![PyPI](https://img.shields.io/pypi/v/qiskit-hamiltonian-simulation)](https://pypi.org/project/qiskit-hamiltonian-simulation/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/qiskit-hamiltonian-simulation/)

Time evolution `e^(-i t H / hbar)` under potentials `V(x)` and kinetic energies `T(p)`, given as [`python-signals`](https://github.com/BlackWild/qiu/blob/master/shareable-packages/python-signals/README.md) signals, built on the phase circuits of [`qiskit-phase-propagator`](https://github.com/BlackWild/qiu/blob/master/packages/qiskit-phase-propagator/README.md).

## Installation

```sh
pip install qiskit-hamiltonian-simulation
```

## Position and momentum domain

A potential acts diagonally on the position amplitudes. A kinetic energy acts diagonally on the momentum amplitudes, the orthonormal DFT `numpy.fft.fft(psi, norm="ortho")`, which is Qiskit's **inverse** QFT (`time_independent.fourier`). The momentum-domain evolutions thus apply the inverse QFT, the phase and the QFT, in this order.

The momentum signal must live on a Fourier domain axis (e.g. `MomentumAxis` or `AngularWavenumberAxis`) in the `FFT` ordering, whose sample `k` is the momentum of the Fourier basis state `|k>`. Other orderings, and signals on axes of the wrong domain, are rejected.

## Evolutions

`time_independent.direct`, exact for quadratic signals `f`, i.e. `e^(i f)`:

- `PositionDomainEvolutionQuadratic(f)` for `f` on a position axis.
- `MomentumDomainEvolutionQuadratic(f, fourier_method)` for `f` on a momentum axis.

For the evolution under `V` or `T` over the time `t`, pass `f = (-t / hbar) * V`; scaling a `QuadraticSignal` keeps it quadratic.

`time_independent.sample_based`, for arbitrary real signals of one sign, sampled or algebraic:

- `PotentialEvolutionSampleBased(V, t, hbar, max_delta, state_preparation_method)`
- `KineticEvolutionSampleBased(T, t, hbar, max_delta, state_preparation_method, fourier_method)`

Both apply `e^(-i t V / hbar)` or `e^(-i t T / hbar)` with the sample-based phase propagator, sliced into phases of at most `max_delta`, and expose its `num_of_cycles`.

All synthesis methods default to `GATE`, leaving the synthesis to Qiskit when transpiling; see the caveat on the state preparation in [`qiskit-phase-propagator`](https://github.com/BlackWild/qiu/blob/master/packages/qiskit-phase-propagator/README.md#sample-based-phases).

## Usage

```python
import numpy as np
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)

x_axis = PositionAxis(size=16, delta_x=0.5, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=1.0)  # FFT ordering

# free evolution of a particle of mass 1 over the time t = 0.2
t, mass = 0.2, 1.0
kinetic = QuadraticSignal(p_axis, alpha=1 / (2 * mass))
evolution = MomentumDomainEvolutionQuadratic((-t / 1.0) * kinetic)

psi = Statevector(
    np.exp(-(x_axis.values**2)) / np.linalg.norm(np.exp(-(x_axis.values**2)))
)
out = psi.evolve(evolution)
expected = np.fft.ifft(
    np.exp(-1j * t * kinetic.data) * np.fft.fft(psi.data, norm="ortho"), norm="ortho"
)
assert np.allclose(out.data, expected)
```

## Examples

- `examples/free_space_double_slit.py`: The paraxial diffraction of a double slit over 1 km, with the direct propagator on 10 qubits, checked against NumPy's FFT.

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/qiskit-hamiltonian-simulation/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages/qiskit-hamiltonian-simulation/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-hamiltonian-simulation
```
