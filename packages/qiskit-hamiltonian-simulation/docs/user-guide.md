# User Guide

The package holds the evolutions under time-independent Hamiltonians in the subpackage `time_independent`:

| module         | contents                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------- |
| `direct`       | exact evolutions under quadratic signals, in the position or the momentum domain                   |
| `sample_based` | evolutions under arbitrary potentials and kinetic energies of one sign, with the sample-based phase propagator |
| `fourier`      | the changes of basis between the position and the momentum domain, and the checks of the axes       |

The signals and axes are those of [`python-signals`](../../python-signals/), the phase circuits those of [`qiskit-phase-propagator`](../../qiskit-phase-propagator/), and the QFT circuits and the `SynthesisMethod` those of [`qiskit-encore`](../../qiskit-encore/).

## Sign conventions

The evolution over the time `t` under a Hamiltonian `H` is `e^(-i t H / hbar)`. Its two kinds of circuits take their arguments differently:

- The direct evolutions apply `e^(i f)` for a given quadratic signal `f`. For the evolution under a potential `V` or a kinetic energy `T`, pass `f = (-t / hbar) * V` or `f = (-t / hbar) * T`. Scaling a `QuadraticSignal` by a number keeps it a `QuadraticSignal`, so the scaled signal is still accepted.
- The sample-based evolutions take `V` or `T` with `t` and `hbar`, and apply `e^(-i t V / hbar)` or `e^(-i t T / hbar)`, i.e. the phase `e^(i f)` of the signal `f = (-t / hbar) * V`, internally.

`hbar` is in the units of choice, and must be the same as the one of the momentum axis, e.g. `MomentumAxis.from_position_axis(x_axis, hbar=hbar)`, whose spacing is `2 pi hbar / (N delta_x)`. A negative `t` evolves backwards in time.

## Position and momentum domain

A potential `V(x)` is diagonal in the position basis: the basis state `|k>` of the register holds the sample `k` of a position axis, and its amplitude is multiplied by `e^(-i t V(x_k) / hbar)`.

A kinetic energy `T(p)` is diagonal in the momentum basis. For a register of `n` qubits, `N = 2**n`, the momentum amplitudes are the orthonormal discrete Fourier transform of the position amplitudes,

```text
phi(p_k) = sum_j e^(-2 pi i j k / N) psi(x_j) / sqrt(N),
```

i.e. `numpy.fft.fft(psi, norm="ortho")`. Qiskit's QFT maps `|j>` to `sum_k e^(+2 pi i j k / N) |k> / sqrt(N)`, so on the amplitudes it is NumPy's orthonormal **inverse** DFT, and the momentum amplitudes are obtained by the **inverse** QFT:

- [`to_momentum_basis(n, method)`][qiskit_hamiltonian_simulation.time_independent.fourier.to_momentum_basis] is the inverse QFT, `numpy.fft.fft(psi, norm="ortho")` on the amplitudes.
- [`to_position_basis(n, method)`][qiskit_hamiltonian_simulation.time_independent.fourier.to_position_basis] is the QFT, `numpy.fft.ifft(psi, norm="ortho")` on the amplitudes.

The momentum-domain evolutions apply the inverse QFT, the phase and the QFT, in this order. With the transforms swapped, the phase would be applied at `-p` instead of `p`, which only goes unnoticed for kinetic energies symmetric in `p`.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_hamiltonian_simulation.time_independent.fourier import (
    to_momentum_basis,
    to_position_basis,
)

rng = np.random.default_rng(3)
amplitudes = rng.normal(size=8) + 1j * rng.normal(size=8)
psi = Statevector(amplitudes / np.linalg.norm(amplitudes))

for method in SynthesisMethod:
    momentum = psi.evolve(to_momentum_basis(3, method))
    assert np.allclose(momentum.data, np.fft.fft(psi.data, norm="ortho"))
    assert np.allclose(momentum.evolve(to_position_basis(3, method)).data, psi.data)
```

### Why the momentum axis must be in the FFT ordering

After the inverse QFT, the basis state `|k>` holds the DFT coefficient `k`, whose momentum is `p_k = index[k] * delta_p` with the indices of `numpy.fft.fftfreq(N) * N`: `0, 1, ..., N/2 - 1, -N/2, ..., -1`. These are exactly the indices of the `FFT` ordering of `python-signals`, so the sample `k` of a signal on an `FFT`-ordered momentum axis is the value at the momentum of `|k>`. On a `CENTERED` or `NATURAL` axis, the samples would be applied to the wrong basis states.

[`require_momentum_domain_axis`][qiskit_hamiltonian_simulation.time_independent.fourier.require_momentum_domain_axis] therefore raises a `ValueError` unless the axis is a Fourier domain axis (`MomentumAxis`, `AngularWavenumberAxis` or `SpatialFrequencyAxis`) in the `FFT` ordering. The conjugate axes created by `from_position_axis` are in this ordering by default, unless `keep_ordering=True`. [`require_position_domain_axis`][qiskit_hamiltonian_simulation.time_independent.fourier.require_position_domain_axis] raises unless the axis is a position axis. The checks do not compare the momentum axis with a position axis: create it with `from_position_axis` so that its spacing is the one of the DFT.

The ordering of the position axis is free. The DFT sums over the array positions `j`, not the positions `x_j`, so for a position axis starting at `x_0 != 0`, e.g. a `CENTERED` one, the DFT coefficient `k` differs from the momentum amplitude at `p_k` by the phase `e^(i p_k x_0 / hbar)`. The phases cancel between the two transforms, since the kinetic phase is diagonal in between. For the `FFT` ordering, the DFT is periodic in `j`, so the negative positions are summed correctly as they are.

!!! note "Periodic boundaries"
    The DFT makes the position window periodic: a wave packet leaving it at one end reenters at the other. The momenta are limited to `|p| <= pi hbar / delta_x`, so the sampling must resolve the fastest oscillations of the state.

## Direct evolutions

`time_independent.direct` applies `e^(i f)` exactly for a `QuadraticSignal` `f`, with the direct phase circuit of `qiskit-phase-propagator` on a register `psi` of `n` qubits:

- [`PositionDomainEvolutionQuadratic(f)`][qiskit_hamiltonian_simulation.time_independent.direct.PositionDomainEvolutionQuadratic] for `f` on a position axis, e.g. a harmonic potential `f = (-t / hbar) * QuadraticSignal(x_axis, m omega^2 / 2)`.
- [`MomentumDomainEvolutionQuadratic(f, fourier_method)`][qiskit_hamiltonian_simulation.time_independent.direct.MomentumDomainEvolutionQuadratic] for `f` on a momentum axis, e.g. a free particle `f = (-t / hbar) * QuadraticSignal(p_axis, 1 / (2 m))`, between `to_momentum_basis` and `to_position_basis`.

Both are unitary circuits without measurements, so they run with `Statevector.evolve`, and store the signal as `quadratic_signal`. The `fourier_method` chooses how the transforms are represented: `GATE` (default) a Qiskit `QFTGate`, `DECOMPOSED` Hadamard, controlled phase and swap gates, `DENSE` a unitary matrix.

A sum of signals, e.g. `V + 1` or a shifted parabola, is no longer a `QuadraticSignal`. A constant only adds a global phase; other polynomials up to the power 3 can be composed from the circuits of `qiskit_phase_propagator.direct.polynomial_phase_circuit`, one per monomial.

## Sample-based evolutions

`time_independent.sample_based` applies the evolution under an arbitrary real signal of one sign, a `Signal` or an `AlgebraicSignal`, with the `QuadraticSignalSampleBasedPhasePropagator` of `qiskit-phase-propagator` on the signal `(-t / hbar) * V`:

- [`PotentialEvolutionSampleBased(V, t, hbar, max_delta, state_preparation_method)`][qiskit_hamiltonian_simulation.time_independent.sample_based.PotentialEvolutionSampleBased] for a potential on a position axis.
- [`KineticEvolutionSampleBased(T, t, hbar, max_delta, state_preparation_method, fourier_method)`][qiskit_hamiltonian_simulation.time_independent.sample_based.KineticEvolutionSampleBased] for a kinetic energy on an `FFT`-ordered Fourier domain axis, between `to_momentum_basis` and `to_position_basis` on the `psi` register.

The phase `-t sum_k V(x_k) / hbar` is sliced into `num_of_cycles = ceil(t |sum_k V(x_k)| / (hbar max_delta))` equal phases of magnitude at most `max_delta`, one cycle each. The circuits have the registers of the propagator: `psi` (qubits `0, ..., n-1`), `phi` (qubits `n, ..., 2n-1`) and the classical `success_flag` of `n` bits, which is 0 if all cycles succeeded. The protocol, its closed-form cycle map, its success probability and how to run it on Aer are explained in the [User Guide of qiskit-phase-propagator](../../qiskit-phase-propagator/user-guide/).

The `state_preparation_method` of `|phi>` defaults to `GATE`, Qiskit's `StatePreparation`, which is unreliable for the nearly uniform `|phi>` of smooth signals (qiskit 2.2); pass `SynthesisMethod.DECOMPOSED` for a robust synthesis, or `DENSE` on few qubits.

The sample-based evolutions are approximate, with an error of order `t |sum_k V(x_k)| max_delta / hbar`, and probabilistic: they succeed with a probability of at least `1 - t |sum_k V(x_k)| max_delta / (4 hbar)`.

## Combining evolutions

The package applies one term of the Hamiltonian at a time. For `H = T + V`, whose terms do not commute, compose the terms by operator splitting, e.g. the second-order (Strang) splitting over `steps` steps of `dt = t / steps`,

```text
e^(-i t H / hbar) ~ (e^(-i dt V / (2 hbar)) e^(-i dt T / hbar) e^(-i dt V / (2 hbar)))^steps,
```

with a global error of order `dt^2` for a fixed time `t`. The direct evolutions compose on a register of `n` qubits like any circuit. To compose sample-based evolutions, build the circuit from the registers of `qiskit_phase_propagator.sample_based.propagator_registers(n)` and compose the direct evolutions onto its `psi` register; guard each sample-based evolution after the first with `circuit.if_test((success_flag, 0))`, so that a later success does not overwrite an earlier failure. See the [Examples](examples.md#a-wave-packet-at-a-potential-step).

## Pitfalls

- The axes must have `2**n` samples, and the momentum axis the `FFT` ordering; a potential on a momentum axis, or a kinetic energy on a position axis, raises a `ValueError`.
- The sample-based evolutions need a signal of one sign, and a vanishing phase raises a `ValueError`, e.g. for `t = 0`. A potential of mixed signs can be shifted by a constant, `e^(-i t (V + c) / hbar) = e^(-i t c / hbar) e^(-i t V / hbar)`, at the price of a global phase and more cycles.
- The number of cycles grows with the sum of the samples, i.e. with the number of samples for a potential of fixed magnitude, and with the evolution time.
- Transpile the sample-based circuits at an optimization level of at most 1 when comparing amplitudes: from level 2, Qiskit drops gates close to the identity.
