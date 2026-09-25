# User Guide

The package has four modules:

| module                | contents                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `qubit_encoding`      | how an axis of `2**n` samples is represented by `n` qubits, and the bit weights of its indices |
| `direct`              | exact phase circuits `e^(i alpha x^power)` for monomials up to the power 3                  |
| `sample_based`        | the sample-based protocol for arbitrary real signals of one sign, and its circuits           |
| `sample_based_manual` | the statevector simulation of the sample-based protocol, post-selected on success            |

The signals, axes and index orderings are those of [`python-signals`](../../python-signals/); the state preparations and the `SynthesisMethod` are those of [`qiskit-encore`](../../qiskit-encore/).

## Encoding of an axis in qubits

An axis of `2**n` samples is represented by `n` qubits, and [`num_qubits_of`][qiskit_phase_propagator.qubit_encoding.num_qubits_of] returns `n`. It raises a `ValueError` for any other size, including a single sample (`n = 0`).

The sample at array position `k` is the basis state `|k>`, with `k = sum_i 2^i x_i` in Qiskit's little-endian order: qubit 0 holds the least significant bit. The basis state thus encodes the integer index `axis.index[k]`, which depends on the index ordering of the axis. [`bit_weights`][qiskit_phase_propagator.qubit_encoding.bit_weights] returns the weights `w_i` such that the encoded integer is `sum_i w_i x'_i`:

| ordering   | encoded integer of the bits `x_(n-1) ... x_0`            | weights for `n = 3` | MSB flipped |
| ---------- | -------------------------------------------------------- | ------------------- | ----------- |
| `NATURAL`  | unsigned, `sum_i 2^i x_i`                                | `[1, 2, 4]`         | no          |
| `FFT`      | two's complement, `-2^(n-1) x_(n-1) + sum_(i<n-1) 2^i x_i` | `[1, 2, -4]`        | no          |
| `CENTERED` | two's complement of the bits with the top one flipped    | `[1, 2, -4]`        | yes         |

For the `CENTERED` ordering, `x'_(n-1) = 1 - x_(n-1)` is the flipped most significant bit, and `x'_i = x_i` otherwise; [`is_msb_flipped`][qiskit_phase_propagator.qubit_encoding.is_msb_flipped] tells which orderings need it. Circuits built from the weights flip the most significant qubit with an `X` gate before and after applying them. The weights reproduce the indices of every ordering:

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering, IntegerAxis
from qiskit_phase_propagator.qubit_encoding import (
    bit_weights,
    is_msb_flipped,
    num_qubits_of,
)

for ordering in IndexOrdering:
    axis = IntegerAxis(8, ordering)
    num_qubits = num_qubits_of(axis)
    bits = (np.arange(8)[:, None] >> np.arange(num_qubits)) & 1  # bits[k, i] = x_i
    if is_msb_flipped(ordering):
        bits[:, -1] ^= 1
    assert np.array_equal(bits @ bit_weights(num_qubits, ordering), axis.index)
```

The physical value of the sample `k` is `axis.index[k] * axis.period`, so a signal `alpha x^power` is `effective_alpha * index^power` in terms of the encoded integers, with `effective_alpha = alpha * period^power` (see `PolynomialSignal.effective_alpha` in `python-signals`).

## Direct phases

[`polynomial_phase_circuit(signal)`][qiskit_phase_propagator.direct.polynomial_phase_circuit] returns the diagonal circuit mapping `|k>` to `e^(i signal(x_k)) |k>` for a `PolynomialSignal` `alpha x^power` with `power <= 3`, e.g. a `QuadraticSignal`. The phase is exact, global phase included.

The circuit expands the power of the encoded integer `x = sum_i w_i x_i` into products of bits, using `x_i^2 = x_i`:

- `x = sum_i w_i x_i`: one phase gate per qubit ([`Order1DirectPhase`][qiskit_phase_propagator.direct.Order1DirectPhase]).
- `x^2 = sum_i w_i^2 x_i + 2 sum_(j<i) w_i w_j x_i x_j`: in addition, one controlled phase gate per pair of qubits ([`Order2DirectPhase`][qiskit_phase_propagator.direct.Order2DirectPhase]).
- `x^3 = sum_i w_i^3 x_i + 3 sum_(j<i) (w_i^2 w_j + w_i w_j^2) x_i x_j + 6 sum_(k<j<i) w_i w_j w_k x_i x_j x_k`: in addition, one doubly controlled phase gate per triple of qubits ([`Order3DirectPhase`][qiskit_phase_propagator.direct.Order3DirectPhase]).

On `n` qubits, the circuits thus have `n` phase gates, `n (n - 1) / 2` controlled phase gates for the powers 2 and 3, `n (n - 1) (n - 2) / 6` multi-controlled phase gates for the power 3, and two `X` gates for the `CENTERED` ordering. The power 0, a constant signal, is a circuit without gates whose `global_phase` is `alpha`. Higher powers raise a `NotImplementedError`.

The classes derive from [`DirectPhase`][qiskit_phase_propagator.direct.DirectPhase] and can be used without a signal, e.g. `Order2DirectPhase(num_qubits, coef, ordering)` applies `e^(i coef index^2)` in terms of the integer indices; the ordering may be given as its raw value, e.g. `"centered"`. They store the `coef`, the `ordering` and the class attribute `exponent`, which is not called `power` since that would shadow `QuantumCircuit.power`. [`DIRECT_PHASES`][qiskit_phase_propagator.direct.DIRECT_PHASES] maps each exponent to its class.

!!! note "Polynomials"
    Only monomials are supported. A polynomial is the product of the phases of its monomials, i.e. the composition of their circuits, and a constant term is a global phase. For example, `alpha (x - x0)^2 = alpha x^2 - 2 alpha x0 x + alpha x0^2` is a quadratic, a linear and a constant phase, see the [Examples](examples.md#a-shifted-lens).

## Sample-based phases

The sample-based protocol applies `e^(i f(x))` for an arbitrary real signal `f` of one sign, a `Signal` or an `AlgebraicSignal` (`SampledSignal`), at the price of a second register of `n` qubits and a probabilistic success.

### The decomposition f = alpha |phi|^2

[`sample_based_decomposition(signal)`][qiskit_phase_propagator.sample_based.sample_based_decomposition] splits the signal into the sum `alpha` of its samples and the normalized state `|phi> = sqrt(f / alpha)`, such that `f = alpha |phi|^2` sample by sample. A non-positive signal has a negative `alpha`, and `|phi>` is real and non-negative in both cases. It raises a `ValueError` if the samples have mixed signs, if they all vanish, or if the data has non-zero imaginary parts; complex data with vanishing imaginary parts is accepted.

### Slicing alpha into deltas

[`slice_alpha_to_deltas_evenly(alpha, max_delta)`][qiskit_phase_propagator.sample_based.slice_alpha_to_deltas_evenly] returns the fewest equal phases `delta` of magnitude at most `max_delta` that sum up to `alpha`: `ceil(|alpha| / max_delta)` of them, each `alpha / ceil(|alpha| / max_delta)`, of the sign of `alpha`. `max_delta` must be positive.

### One cycle

Each cycle acts on the register `psi` holding the state and a register `phi` of the same size, starting in `|0...0>`:

1. Prepare `|phi>` in the `phi` register.
2. Apply `e^(i delta)` to the basis states `|j>|l>` with `j == l`, i.e. where both registers agree: [`partial_phase_circuit(delta, n)`][qiskit_phase_propagator.sample_based.partial_phase_circuit]. It flags the agreeing bits in place with `2n` open-controlled `CX` gates, applies one multi-controlled phase gate, and restores `psi`. Its unitary is diagonal, with the entry `e^(i delta [j == l])` at the index `l * 2**n + j` ([`partial_phase_diagonal`][qiskit_phase_propagator.sample_based.partial_phase_diagonal]).
3. Un-prepare `|phi>` and measure the `phi` register. The cycle succeeds if it is measured in `|0...0>`.

### Closed-form cycle map and success probability

Writing `w_j = |phi_j|^2`, a successful cycle maps the amplitudes of `psi` to

```text
psi_j  ->  psi_j (1 + (e^(i delta) - 1) w_j) / sqrt(P)
```

with the success probability

```text
P = sum_j |psi_j|^2 |1 + (e^(i delta) - 1) w_j|^2
  = 1 - 2 (1 - cos delta) sum_j |psi_j|^2 w_j (1 - w_j).
```

Since `1 + (e^(i delta) - 1) w_j = e^(i delta w_j) (1 - delta^2 w_j (1 - w_j) / 2 + O(delta^3))`, a cycle applies `e^(i delta |phi_j|^2)` up to `O(delta^2)`, and the cycles of all deltas apply `e^(i alpha |phi|^2) = e^(i f)`. The deviation is mostly a non-uniform damping of the amplitudes, of order `|alpha| max_delta` in total: halving `max_delta` doubles the number of cycles and halves the error.

Since `w_j (1 - w_j) <= 1/4` and `2 (1 - cos delta) <= delta^2`, each cycle fails with a probability of at most `delta^2 / 4`, and all `|alpha| / |delta|` cycles succeed with a probability of at least `1 - |alpha| max_delta / 4`, whatever the state `psi`.

## Propagator circuits

The propagators act on the registers returned by [`propagator_registers(n)`][qiskit_phase_propagator.sample_based.propagator_registers]: `psi` on the qubits `0, ..., n-1`, `phi` on the qubits `n, ..., 2n-1`, and the classical register `success_flag` of `n` bits. They use classical control flow (`if_test`, `for_loop`), so they run on simulators and devices supporting dynamic circuits, e.g. Aer, but not with `Statevector.evolve`. All of them expose their `num_of_cycles`.

- [`QuadraticSignalSampleBasedPhasePropagator(signal, max_delta, method)`][qiskit_phase_propagator.sample_based.QuadraticSignalSampleBasedPhasePropagator] is the complete propagator for a signal: it decomposes it, slices `alpha` evenly and prepares `|phi>` with a `PreparableState` of the given `SynthesisMethod`. Despite its name, the signal is not restricted to a `QuadraticSignal`: any real signal of one sign is accepted, sampled or algebraic.
- [`GenericIterativeSampleBasedPhasePropagatorWithConstantDelta(delta, number_of_cycles, U_phi, U_phi_dagger)`][qiskit_phase_propagator.sample_based.GenericIterativeSampleBasedPhasePropagatorWithConstantDelta] repeats one cycle in a `for_loop` that breaks at the first failure. The complete propagator above is built on it.
- [`GenericIterativeSampleBasedPhasePropagator(deltas, U_phi, U_phi_dagger, take_snapshot)`][qiskit_phase_propagator.sample_based.GenericIterativeSampleBasedPhasePropagator] unrolls one cycle per delta, each in an `if_test` running only if all previous cycles succeeded, so the deltas may differ. With `take_snapshot=True`, it saves the statevector after each cycle, labeled by the cycle index as a string (Aer only).

The generic propagators take the preparation circuit `U_phi` and its inverse directly, or a `PreparableState` via their `from_state` constructors. After each measurement, the `phi` register is reset: on success it is already `|0...0>`, and after a failure the reset keeps the corrupted output inspectable.

### Running on Aer

To run a propagator, initialize `psi`, compose the propagator, save the statevector and run a single shot. The success flags are the counts, and after a success the `phi` register is back in `|0...0>`, so the first `2**n` amplitudes of the saved statevector are the output of `psi`, exact including their global phase:

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer_encore.simulator import aer_simulator
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)

f = Signal(PositionAxis(4, 1.0, IndexOrdering.FFT), [0.05, 0.1, 0.15, 0.1])
propagator = QuadraticSignalSampleBasedPhasePropagator(
    f, max_delta=0.05, method=SynthesisMethod.DECOMPOSED
)
psi = Statevector.from_label("++")

circuit = QuantumCircuit(*propagator.qregs, *propagator.cregs)
circuit.initialize(psi, propagator.qregs[0])
circuit.compose(propagator, inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
compiled = transpile(circuit, simulator, optimization_level=1)
result = simulator.run(compiled, shots=1).result()

succeeded = result.get_counts().int_outcomes().get(0) == 1
output = np.asarray(result.get_statevector().data)[:4]
assert succeeded
assert abs(np.vdot(output, np.exp(1j * f.data) * psi.data)) ** 2 > 0.9999
```

!!! warning "Transpiling"
    Transpile at an optimization level of at most 1. From level 2, Qiskit removes gates it deems equivalent to the identity, e.g. rotations by angles of `1e-7`, which changes the amplitudes by as much and spoils the comparison with the closed form.

!!! warning "Composing propagators"
    Compose the propagators onto registers named like those of `propagator_registers`, e.g. a circuit built from `propagator.qregs` and `propagator.cregs`. With Qiskit 2.2, composing them onto a classical register of another name leaves the condition inside the loop referring to a different register, and Aer fails to run the circuit. The loop of `GenericIterativeSampleBasedPhasePropagatorWithConstantDelta`, and thus of `QuadraticSignalSampleBasedPhasePropagator`, starts regardless of the flags, so several of them composed onto the same `success_flag` overwrite each other's flags; guard each later one with `circuit.if_test((success_flag, 0))` to keep the first failure visible.

## Statevector simulation

`sample_based_manual` simulates the same protocol on Qiskit statevectors, keeping the successful outcome of each cycle and renormalizing it instead of measuring. It applies the partial phase as its diagonal, which is much faster than simulating its multi-controlled phase gate, and needs no dynamic circuits:

- [`phase_propagation_cycle(psi, delta, phi)`][qiskit_phase_propagator.sample_based_manual.phase_propagation_cycle] simulates one cycle for a `PreparableState` `phi` and returns the normalized output and the success probability `P`.
- [`phase_propagate_state(psi_in, deltas, phi)`][qiskit_phase_propagator.sample_based_manual.phase_propagate_state] and [`phase_propagate_state_with_constant_delta(psi_in, delta, num_cycles, phi)`][qiskit_phase_propagator.sample_based_manual.phase_propagate_state_with_constant_delta] simulate one successful cycle per delta.
- [`phase_propagate_state_with_arbitrary_signal(psi_in, signal, max_delta, method)`][qiskit_phase_propagator.sample_based_manual.phase_propagate_state_with_arbitrary_signal] is the counterpart of `QuadraticSignalSampleBasedPhasePropagator`: it decomposes and slices the signal the same way.

The simulation still builds and simulates the preparation circuits of `|phi>`, so the `SynthesisMethod` matters here as well; `DENSE` is the fastest on few qubits.

## Synthesis of the state preparation

The `method` of the propagators and of the simulation defaults to `GATE`, a Qiskit `StatePreparation` synthesized when transpiling. Qiskit's synthesis is unreliable for the nearly uniform `|phi>` of smooth signals (qiskit 2.2): transpiling can fail in its two-qubit decomposition, and it can prepare wrong states (see [`qiskit-encore`](../../qiskit-encore/)). Pass `SynthesisMethod.DECOMPOSED` for the Möttönen synthesis of `qiskit-encore`, which is numerically robust, or `SynthesisMethod.DENSE` for exact results on few qubits.

## Pitfalls

- The axis must have `2**n` samples with `n >= 1`, for the direct and the sample-based phases alike.
- The direct phases need a `PolynomialSignal` of power at most 3; arithmetic other than scaling, e.g. `lens + 1`, returns a plain `AlgebraicSignal` without a `power`.
- The sample-based phases need a signal of one sign. A signal of mixed signs can be shifted by a constant, which only changes the global phase, `e^(i (f + c)) = e^(i c) e^(i f)`; the shift increases `|alpha|` and thus the number of cycles, and makes `|phi>` more uniform.
- A vanishing signal raises a `ValueError`, e.g. a sample-based phase scaled by a time of 0.
- The number of cycles grows as `|alpha| / max_delta`, with `alpha` the sum of all samples, i.e. with the number of samples for a signal of fixed magnitude.
