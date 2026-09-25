# qiskit-hamiltonian-simulation

Circuits of the time evolution `e^(-i t H / hbar)` under a time-independent potential `V(x)` or kinetic energy `T(p)`, given as [`python-signals`](../python-signals/) signals on axes of `2**n` samples. A potential acts diagonally on the position amplitudes of a register, a kinetic energy on its momentum amplitudes, between Fourier transforms.

Quadratic potentials and kinetic energies, e.g. harmonic potentials, free particles or paraxial diffraction, are applied exactly by the direct phase circuits of [`qiskit-phase-propagator`](../qiskit-phase-propagator/). Arbitrary ones of one sign, sampled or algebraic, are applied by its sample-based phase propagator, sliced into small phases. The Fourier transforms are the QFT circuits of [`qiskit-encore`](../qiskit-encore/).

In the monorepo, the direct momentum-domain evolution is the free-space propagator of the Qiskit backend of the lens experiments in [`wave_optics_propagation`](../wave_optics_propagation/), which runs the simulation loop of [`python-wave-optics`](../python-wave-optics/).

## Installation

```sh
pip install qiskit-hamiltonian-simulation
```

Inside the monorepo, it is installed with the other workspace packages by `uv sync --all-packages`.

## Quick start

The free evolution of a Gaussian wave packet of mass 1 over the time `t = 0.2`, applied in the momentum domain and compared with NumPy's FFT:

```python
import numpy as np
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)

hbar, mass, t = 1.0, 1.0, 0.2
x_axis = PositionAxis(size=16, delta_x=0.5, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)  # FFT ordering

kinetic = QuadraticSignal(p_axis, alpha=1 / (2 * mass))
evolution = MomentumDomainEvolutionQuadratic((-t / hbar) * kinetic)

packet = np.exp(-(x_axis.values**2))
psi = Statevector(packet / np.linalg.norm(packet))
out = psi.evolve(evolution)

momentum_amplitudes = np.fft.fft(psi.data, norm="ortho")
expected = np.fft.ifft(
    np.exp(-1j * t / hbar * kinetic.data) * momentum_amplitudes, norm="ortho"
)
assert np.allclose(out.data, expected)
```

## Where next

- The [User Guide](user-guide.md) explains the sign conventions, the position and momentum domains and their Fourier transforms, the direct and the sample-based evolutions, and how to combine them.
- The [Examples](examples.md) work through a free wave packet, a harmonic oscillator by operator splitting, a potential step and a relativistic kinetic energy.
- The [API Reference](reference/qiskit_hamiltonian_simulation/index.md) documents every module: [`time_independent.direct`](reference/qiskit_hamiltonian_simulation/time_independent/direct.md), [`time_independent.sample_based`](reference/qiskit_hamiltonian_simulation/time_independent/sample_based.md) and [`time_independent.fourier`](reference/qiskit_hamiltonian_simulation/time_independent/fourier.md).
