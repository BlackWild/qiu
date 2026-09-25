# User Guide

`qiu-quantum-computing` computes with the amplitudes of a quantum computer, without reference to the physical dynamics they may simulate. Its modules:

| module                                                                                                                  | contents                                                     |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [`state_preparation`][qiu_quantum_computing.state_preparation], [`preparable_state`][qiu_quantum_computing.preparable_state] | circuits preparing a state from the all-zero state, and back |
| [`qft`][qiu_quantum_computing.qft]                                                                                      | the quantum Fourier transform and its inverse                |
| [`uniformly_controlled_rotation`][qiu_quantum_computing.uniformly_controlled_rotation]                                  | multiplexed `ry` and `rz` rotations                          |
| [`phase_propagator`][qiu_quantum_computing.phase_propagator]                                                            | diagonal phase operators `e^(i f(x))` of sampled signals     |

State preparation and the QFT are represented in their circuits as chosen by the `SynthesisMethod` of [qiu-qiskit-encore](../qiu-qiskit-encore/index.md): as a dense unitary, a high-level Qiskit gate or a decomposed circuit, see its [User Guide](../qiu-qiskit-encore/user-guide.md#synthesis-methods). The uniformly controlled rotations are always decomposed into elementary gates. The signals, axes and index orderings of the phase propagator are those of [qiu-signals](../qiu-signals/index.md).

## Conventions

### Qubit ordering

The package follows Qiskit's little-endian convention throughout: the amplitude at index `j = sum_q b_q 2**q` of a statevector belongs to the basis state in which qubit `q` is in `|b_q>`. Qubit 0 is thus the least significant bit of the index. The QFT encodes its integers in the same order, and the controls of a uniformly controlled rotation select its angle by the same rule.

### Global phase

Every circuit of this package implements its operation exactly, *including the global phase*. A state preparation circuit `U` satisfies `U|0...0> = |state>`, not merely `U|0...0> = e^(i phi)|state>`. Results can therefore be compared by their amplitudes, e.g. with `numpy.allclose` or the equality of Qiskit's `Statevector`, rather than only up to a phase with `Statevector.equiv`. The global phase matters as soon as the circuit is controlled, e.g. in phase estimation or in the Hadamard test.

### Input states

The state preparation functions and [`PreparableState`][qiu_quantum_computing.preparable_state.PreparableState] accept a Qiskit `Statevector` or anything NumPy converts to an array of amplitudes. They validate it with `validated_statevector` of [qiu-qiskit-encore](../qiu-qiskit-encore/index.md), which returns a copy as a complex `Statevector` and raises a `ValueError` unless:

- the dimension is a power of 2 of at least one qubit, i.e. `2, 4, 8, ...`, and
- the state is normalized, as checked by `Statevector.is_valid`, i.e. up to Qiskit's tolerances `Statevector.atol` and `Statevector.rtol`.

States are never normalized silently: normalize them yourself, e.g. with `state / numpy.linalg.norm(state)`.

## State preparation

[`state_preparation_circuit(state, *, method, inverse)`][qiu_quantum_computing.state_preparation.state_preparation_circuit] returns a circuit on `n` qubits for a state of `2**n` amplitudes. With `inverse=False` (the default), it maps `|0...0>` to the state; with `inverse=True`, it maps the state back to `|0...0>`, as needed for un-computing a state or measuring an overlap. The method defaults to `GATE`.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

state = np.array([0.6, 0, 0, 0.8j])
for method in SynthesisMethod:
    circuit = state_preparation_circuit(state, method=method)
    assert np.allclose(Statevector(circuit).data, state)

    inverse = state_preparation_circuit(state, method=method, inverse=True)
    assert np.allclose(Statevector(state).evolve(inverse).data, [1, 0, 0, 0])
```

The three methods are backed by the following syntheses.

### `GATE`: Qiskit's `StatePreparation`

The circuit holds a single Qiskit `StatePreparation` gate, named `state_preparation`, with `inverse=True` passed on to it. Qiskit synthesizes it, when transpiling or when a `Statevector` is built from the circuit, by its isometry synthesis. For generic states, this synthesis needs the fewest CNOT gates of the three methods.

!!! warning "Qiskit's synthesis prepares wrong states for some inputs"
    Qiskit's isometry synthesis prepares wrong states, with fidelity 0, when two of its intermediate single-qubit gates are close but not equal. The tests of this package document this with a state of 3 qubits found by `hypothesis`, on which it fails in qiskit 2.2 to 2.5. Whether the two gates are close depends on the rounding of the linear algebra: it fails with Apple's Accelerate as the LAPACK of NumPy, e.g. on macOS, and not with OpenBLAS, e.g. on Linux. With Accelerate, the test is marked as an expected failure, and alerts once Qiskit fixes it.

!!! warning "Transpiling can fail for nearly uniform states"
    With qiskit 2.2, the synthesis of the `StatePreparation` of a nearly uniform state, such as the state `sqrt(f / sum(f))` of a smooth signal `f`, can fail: Qiskit's decomposition of its uniformly controlled gates produces a single-qubit matrix that it then rejects as not unitary. Transpiling raises a `TranspilerError` (`HighLevelSynthesis is unable to synthesize "state_preparation"`), and building a `Statevector` of the circuit a `ValueError`. Which states are affected is hard to predict. This is why the [sample-based phases](#sample-based-phases) of this package and the circuits of [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/index.md) are tested with `DECOMPOSED`.

Use `DECOMPOSED` wherever these failures matter, and `GATE` where a hardware-aware synthesis by Qiskit is more important than robustness.

### `DECOMPOSED`: uniformly controlled rotations

[`decomposed_state_preparation(state)`][qiu_quantum_computing.state_preparation.decomposed_state_preparation] implements the synthesis of Möttönen et al., "Transformation of quantum states using uniformly controlled rotations" (2005). It works in two stages:

1. **Magnitudes.** For each target qubit `t`, from the most significant one down, a uniformly controlled `ry` rotation, controlled by the more significant qubits `t+1, ..., n-1`, rotates the target into the conditional distribution of its bit. After this stage, the circuit prepares the magnitudes `|state|`.
2. **Phases.** For each target qubit `t`, from the least significant one up, the phases of each pair of amplitudes differing in the bit of `t` are split into their difference, applied by a uniformly controlled `rz` rotation, and their mean, left to the more significant qubits. The mean left after the last qubit becomes the global phase of the circuit.

The circuit contains only `ry`, `rz` and `cx` gates, and has the following properties:

- It uses at most `2**(n+1) - 4` CNOT gates for `n` qubits, i.e. `2**n - 2` per stage.
- For real non-negative states, the phase stage has only zero angles, so the circuit contains no `rz` gates and has global phase 0: at most `2**n - 2` CNOT gates.
- Rotations with zero angle are omitted, and rotations independent of their controls need no CNOT gates, so structured states, e.g. with zeros or product structure, get shorter circuits.
- The phases of zero amplitudes are ignored, so zeros need no special treatment.
- All angles come from `arctan2` and sums, without eigendecompositions, so the synthesis is numerically robust for any state, including the ones on which Qiskit's synthesis fails.

With `inverse=True`, `state_preparation_circuit` returns the inverse of this circuit, i.e. the gates in reverse order with negated angles.

### `DENSE`: a Householder reflection

[`state_preparation_unitary(state)`][qiu_quantum_computing.state_preparation.state_preparation_unitary] returns a Qiskit `Operator` whose first column is the state: `-e^(i theta) H`, where `H` is the Householder reflection swapping `|0...0>` and `-e^(-i theta)|state>`, and `theta` is the phase of the first amplitude. The reflection vector has norm at least 1, so the construction is numerically stable for every state, and it is exact. The `DENSE` circuit holds this operator as a single `unitary` gate labeled `state_preparation`, or its adjoint labeled `state_preparation_dg` for `inverse=True`. The matrix has `4**n` entries, so it is meant for few qubits.

The other columns of the unitary are fixed by the reflection and carry no meaning: only `U|0...0>` is specified.

## Preparable states

[`PreparableState(statevector, method)`][qiu_quantum_computing.preparable_state.PreparableState] bundles a state with its synthesis method, for code that prepares and un-prepares the same state repeatedly, e.g. in every cycle of a protocol:

- `statevector` is the validated copy of the given state, and `method` the synthesis method, `GATE` by default. Raw method values are converted, as everywhere.
- `num_qubits` is the number of qubits of the state.
- `circuit` and `inverse_circuit` are the circuits of `state_preparation_circuit` with `inverse=False` and `inverse=True`. They are built on first access and cached.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.preparable_state import PreparableState

phi = PreparableState([0.6, 0.8j], method="decomposed")
assert phi.circuit is phi.circuit  # built once
assert np.allclose(Statevector(phi.circuit).data, [0.6, 0.8j])
assert np.allclose(phi.statevector.evolve(phi.inverse_circuit).data, [1, 0])
```

A `PreparableState` is a frozen dataclass: its fields cannot be reassigned, and it holds a copy of the amplitudes, so changing the given array later does not affect it. The cached circuits therefore always prepare the state. Two preparable states are equal when their methods are equal and their statevectors are equal by Qiskit's `Statevector` equality, i.e. up to Qiskit's tolerances. Since a Qiskit `Statevector` is not hashable, neither is a `PreparableState`: it cannot be used as a dictionary key or in a set.

The cached circuits are ordinary, mutable `QuantumCircuit`s. Compose them into other circuits rather than modifying them in place, or the cache no longer matches the state.

## Quantum Fourier transform

[`qft_circuit(num_qubits, *, inverse, method)`][qiu_quantum_computing.qft.qft_circuit] returns the QFT on `num_qubits` qubits, at least 1, or its inverse with `inverse=True`:

| method           | the circuit contains                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| `GATE` (default) | a single Qiskit `QFTGate`, or its inverse, synthesized when transpiling       |
| `DECOMPOSED`     | Hadamard, controlled phase and swap gates of Qiskit's `synth_qft_full`         |
| `DENSE`          | a single unitary gate of the matrix of [`qft_matrix`][qiu_quantum_computing.qft.qft_matrix], or its adjoint |

The QFT maps the basis state `|j>` to `sum_k e^(2 pi i j k / N) |k> / sqrt(N)` with `N = 2**n`, where the integers are encoded in little-endian order like the statevector indices. [`qft_matrix(num_qubits)`][qiu_quantum_computing.qft.qft_matrix] returns this matrix, with the entries `e^(2 pi i j k / N) / sqrt(N)`; it is the matrix of Qiskit's `QFTGate`.

!!! note "The NumPy convention"
    The sign of the exponent is positive, so on the amplitudes the QFT is NumPy's orthonormal **inverse** discrete Fourier transform, and the inverse QFT is the forward one:

    | circuit                           | on the amplitudes `psi`             |
    | --------------------------------- | ----------------------------------- |
    | `qft_circuit(n)`                  | `numpy.fft.ifft(psi, norm="ortho")` |
    | `qft_circuit(n, inverse=True)`    | `numpy.fft.fft(psi, norm="ortho")`  |

    The unnormalized transforms of NumPy's default `norm="backward"` differ from these by the factors `sqrt(N)` and `1 / sqrt(N)`.

The swap gates at the end of `synth_qft_full` are part of the transform, so all three methods implement the same unitary, without a reversal of the qubits left to the user.

!!! note "Transpiling the swaps away"
    From optimization level 2, the default of `transpile`, Qiskit may elide the final swaps of the QFT into a relabeling of the qubits, the `final_layout` of the transpiled circuit. Measurements account for it, but a statevector saved by a simulator, e.g. with Aer's `save_statevector`, then has its qubits permuted. Transpile with `optimization_level=1` where the statevector matters, see [qiu-qiskit-aer-encore](../qiu-qiskit-aer-encore/user-guide.md#exact-statevectors-of-transpiled-circuits).

## Uniformly controlled rotations

[`uniformly_controlled_rotation(axis, angles)`][qiu_quantum_computing.uniformly_controlled_rotation.uniformly_controlled_rotation] returns a multiplexed rotation, the building block of the `DECOMPOSED` state preparation. For `2**k` angles, the circuit acts on `k + 1` qubits:

- qubit 0 is the target, and qubits `1, ..., k` are the controls;
- for the controls in the basis state `|c>`, with `c = sum_j c_j 2**j` over the control qubits `j + 1`, the target is rotated by `R_axis(angles[c])`.

The circuit thus implements the block diagonal unitary `diag(R(angles[0]), R(angles[1]), ...)`, with `R` being `ry` or `rz` for the [`RotationAxis`][qiu_quantum_computing.uniformly_controlled_rotation.RotationAxis] `"y"` or `"z"`.

```python
import numpy as np
from qiskit.quantum_info import Operator
from qiu_quantum_computing.uniformly_controlled_rotation import (
    uniformly_controlled_rotation,
)

circuit = uniformly_controlled_rotation("y", [0.0, np.pi])  # target 0, control 1
assert circuit.num_qubits == 2

# the control in |1> rotates the target by pi: |10> -> |11>
state = np.zeros(4)
state[0b10] = 1
assert np.allclose(Operator(circuit).data @ state, [0, 0, 0, 1])
```

The decomposition of Möttönen et al., "Quantum circuits for general multiqubit gates" (2004), computes the rotation angles by a Walsh-Hadamard transform and places them along a Gray code of the controls:

- a multiplexer with `k >= 1` controls uses at most `2**k` rotations and exactly `2**k` CNOT gates, unless its angles do not depend on the controls;
- rotations with zero angle are omitted;
- angles independent of the controls give a single rotation of the target and no CNOT gates, and zero angles the empty circuit.

The angles must be a one-dimensional array of a power of 2 entries, at least one; other shapes and axes other than `"y"` and `"z"` raise a `ValueError`. Like the rest of the synthesis, the decomposition involves no eigendecompositions and is numerically robust for any angles.

## Pitfalls of the building blocks

- `GATE` state preparation inherits Qiskit's synthesis: wrong states for some inputs, and failing transpilation of nearly uniform states in qiskit 2.2. Use `DECOMPOSED` where this matters.
- `DENSE` circuits are exact but grow as `4**n`, and transpile into many CNOT gates. Keep them for references on few qubits.
- The QFT is NumPy's orthonormal *inverse* DFT, not the forward one.
- From optimization level 2, transpiling may elide the swaps of the QFT into a permutation of the qubits, which statevectors saved by a simulator do not undo; use `optimization_level=1` for exact statevectors.
- The circuits include the global phase; do not drop it when composing controlled versions.
- States must be normalized and of a power-of-2 dimension; they are not normalized for you.
- Do not modify the cached circuits of a `PreparableState` in place.

## Phase propagator

The subpackage [`phase_propagator`][qiu_quantum_computing.phase_propagator] synthesizes diagonal phase operators: circuits applying the phase `e^(i f(x))` of a signal `f` to the basis states of a qubit register. It has four modules:

| module                | contents                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `qubit_encoding`      | how an axis of `2**n` samples is represented by `n` qubits, and the bit weights of its indices |
| `direct`              | exact phase circuits `e^(i alpha x^power)` for monomials up to the power 3                  |
| `sample_based`        | the sample-based protocol for arbitrary real signals of one sign, and its circuits           |
| `sample_based_manual` | the statevector simulation of the sample-based protocol, post-selected on success            |

### Encoding of an axis in qubits

An axis of `2**n` samples is represented by `n` qubits, and [`num_qubits_of`][qiu_quantum_computing.phase_propagator.qubit_encoding.num_qubits_of] returns `n`. It raises a `ValueError` for any other size, including a single sample (`n = 0`).

The sample at array position `k` is the basis state `|k>`, with `k = sum_i 2^i x_i` in Qiskit's little-endian order: qubit 0 holds the least significant bit. The basis state thus encodes the integer index `axis.index[k]`, which depends on the index ordering of the axis. [`bit_weights`][qiu_quantum_computing.phase_propagator.qubit_encoding.bit_weights] returns the weights `w_i` such that the encoded integer is `sum_i w_i x'_i`:

| ordering   | encoded integer of the bits `x_(n-1) ... x_0`            | weights for `n = 3` | MSB flipped |
| ---------- | -------------------------------------------------------- | ------------------- | ----------- |
| `NATURAL`  | unsigned, `sum_i 2^i x_i`                                | `[1, 2, 4]`         | no          |
| `FFT`      | two's complement, `-2^(n-1) x_(n-1) + sum_(i<n-1) 2^i x_i` | `[1, 2, -4]`        | no          |
| `CENTERED` | two's complement of the bits with the top one flipped    | `[1, 2, -4]`        | yes         |

For the `CENTERED` ordering, `x'_(n-1) = 1 - x_(n-1)` is the flipped most significant bit, and `x'_i = x_i` otherwise; [`is_msb_flipped`][qiu_quantum_computing.phase_propagator.qubit_encoding.is_msb_flipped] tells which orderings need it. Circuits built from the weights flip the most significant qubit with an `X` gate before and after applying them. The weights reproduce the indices of every ordering:

```python
import numpy as np
from qiu_signals.integer_axis import IndexOrdering, IntegerAxis
from qiu_quantum_computing.phase_propagator.qubit_encoding import (
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

The physical value of the sample `k` is `axis.index[k] * axis.period`, so a signal `alpha x^power` is `effective_alpha * index^power` in terms of the encoded integers, with `effective_alpha = alpha * period^power` (see `PolynomialSignal.effective_alpha` in `qiu-signals`).

### Direct phases

[`polynomial_phase_circuit(signal)`][qiu_quantum_computing.phase_propagator.direct.polynomial_phase_circuit] returns the diagonal circuit mapping `|k>` to `e^(i signal(x_k)) |k>` for a `PolynomialSignal` `alpha x^power` with `power <= 3`, e.g. a `QuadraticSignal`. The phase is exact, global phase included.

The circuit expands the power of the encoded integer `x = sum_i w_i x_i` into products of bits, using `x_i^2 = x_i`:

- `x = sum_i w_i x_i`: one phase gate per qubit ([`Order1DirectPhase`][qiu_quantum_computing.phase_propagator.direct.Order1DirectPhase]).
- `x^2 = sum_i w_i^2 x_i + 2 sum_(j<i) w_i w_j x_i x_j`: in addition, one controlled phase gate per pair of qubits ([`Order2DirectPhase`][qiu_quantum_computing.phase_propagator.direct.Order2DirectPhase]).
- `x^3 = sum_i w_i^3 x_i + 3 sum_(j<i) (w_i^2 w_j + w_i w_j^2) x_i x_j + 6 sum_(k<j<i) w_i w_j w_k x_i x_j x_k`: in addition, one doubly controlled phase gate per triple of qubits ([`Order3DirectPhase`][qiu_quantum_computing.phase_propagator.direct.Order3DirectPhase]).

On `n` qubits, the circuits thus have `n` phase gates, `n (n - 1) / 2` controlled phase gates for the powers 2 and 3, `n (n - 1) (n - 2) / 6` multi-controlled phase gates for the power 3, and two `X` gates for the `CENTERED` ordering. The power 0, a constant signal, is a circuit without gates whose `global_phase` is `alpha`. Higher powers raise a `NotImplementedError`.

The classes derive from [`DirectPhase`][qiu_quantum_computing.phase_propagator.direct.DirectPhase] and can be used without a signal, e.g. `Order2DirectPhase(num_qubits, coef, ordering)` applies `e^(i coef index^2)` in terms of the integer indices; the ordering may be given as its raw value, e.g. `"centered"`. They store the `coef`, the `ordering` and the class attribute `exponent`, which is not called `power` since that would shadow `QuantumCircuit.power`. [`DIRECT_PHASES`][qiu_quantum_computing.phase_propagator.direct.DIRECT_PHASES] maps each exponent to its class.

!!! note "Polynomials"
    Only monomials are supported. A polynomial is the product of the phases of its monomials, i.e. the composition of their circuits, and a constant term is a global phase. For example, `alpha (x - x0)^2 = alpha x^2 - 2 alpha x0 x + alpha x0^2` is a quadratic, a linear and a constant phase, see the [Examples](examples.md#a-shifted-lens).

### Sample-based phases

The sample-based protocol applies `e^(i f(x))` for an arbitrary real signal `f` of one sign, a `Signal` or an `AlgebraicSignal` (`SampledSignal`), at the price of a second register of `n` qubits and a probabilistic success.

#### The decomposition f = alpha |phi|^2

[`sample_based_decomposition(signal)`][qiu_quantum_computing.phase_propagator.sample_based.sample_based_decomposition] splits the signal into the sum `alpha` of its samples and the normalized state `|phi> = sqrt(f / alpha)`, such that `f = alpha |phi|^2` sample by sample. A non-positive signal has a negative `alpha`, and `|phi>` is real and non-negative in both cases. It raises a `ValueError` if the samples have mixed signs, if they all vanish, or if the data has non-zero imaginary parts; complex data with vanishing imaginary parts is accepted.

#### Slicing alpha into deltas

[`slice_alpha_to_deltas_evenly(alpha, max_delta)`][qiu_quantum_computing.phase_propagator.sample_based.slice_alpha_to_deltas_evenly] returns the fewest equal phases `delta` of magnitude at most `max_delta` that sum up to `alpha`: `ceil(|alpha| / max_delta)` of them, each `alpha / ceil(|alpha| / max_delta)`, of the sign of `alpha`. `max_delta` must be positive.

#### One cycle

Each cycle acts on the register `psi` holding the state and a register `phi` of the same size, starting in `|0...0>`:

1. Prepare `|phi>` in the `phi` register.
2. Apply `e^(i delta)` to the basis states `|j>|l>` with `j == l`, i.e. where both registers agree: [`partial_phase_circuit(delta, n)`][qiu_quantum_computing.phase_propagator.sample_based.partial_phase_circuit]. It flags the agreeing bits in place with `2n` open-controlled `CX` gates, applies one multi-controlled phase gate, and restores `psi`. Its unitary is diagonal, with the entry `e^(i delta [j == l])` at the index `l * 2**n + j` ([`partial_phase_diagonal`][qiu_quantum_computing.phase_propagator.sample_based.partial_phase_diagonal]).
3. Un-prepare `|phi>` and measure the `phi` register. The cycle succeeds if it is measured in `|0...0>`.

#### Closed-form cycle map and success probability

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

### Propagator circuits

The propagators act on the registers returned by [`propagator_registers(n)`][qiu_quantum_computing.phase_propagator.sample_based.propagator_registers]: `psi` on the qubits `0, ..., n-1`, `phi` on the qubits `n, ..., 2n-1`, and the classical register `success_flag` of `n` bits. They use classical control flow (`if_test`, `for_loop`), so they run on simulators and devices supporting dynamic circuits, e.g. Aer, but not with `Statevector.evolve`. All of them expose their `num_of_cycles`.

- [`QuadraticSignalSampleBasedPhasePropagator(signal, max_delta, method)`][qiu_quantum_computing.phase_propagator.sample_based.QuadraticSignalSampleBasedPhasePropagator] is the complete propagator for a signal: it decomposes it, slices `alpha` evenly and prepares `|phi>` with a `PreparableState` of the given `SynthesisMethod`. Despite its name, the signal is not restricted to a `QuadraticSignal`: any real signal of one sign is accepted, sampled or algebraic.
- [`GenericIterativeSampleBasedPhasePropagatorWithConstantDelta(delta, number_of_cycles, U_phi, U_phi_dagger)`][qiu_quantum_computing.phase_propagator.sample_based.GenericIterativeSampleBasedPhasePropagatorWithConstantDelta] repeats one cycle in a `for_loop` that breaks at the first failure. The complete propagator above is built on it.
- [`GenericIterativeSampleBasedPhasePropagator(deltas, U_phi, U_phi_dagger, take_snapshot)`][qiu_quantum_computing.phase_propagator.sample_based.GenericIterativeSampleBasedPhasePropagator] unrolls one cycle per delta, each in an `if_test` running only if all previous cycles succeeded, so the deltas may differ. With `take_snapshot=True`, it saves the statevector after each cycle, labeled by the cycle index as a string (Aer only).

The generic propagators take the preparation circuit `U_phi` and its inverse directly, or a `PreparableState` via their `from_state` constructors. After each measurement, the `phi` register is reset: on success it is already `|0...0>`, and after a failure the reset keeps the corrupted output inspectable.

#### Running on Aer

To run a propagator, initialize `psi`, compose the propagator, save the statevector and run a single shot. The success flags are the counts, and after a success the `phi` register is back in `|0...0>`, so the first `2**n` amplitudes of the saved statevector are the output of `psi`, exact including their global phase:

```python
import numpy as np
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiu_signals.signal import Signal
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
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

### Statevector simulation

`sample_based_manual` simulates the same protocol on Qiskit statevectors, keeping the successful outcome of each cycle and renormalizing it instead of measuring. It applies the partial phase as its diagonal, which is much faster than simulating its multi-controlled phase gate, and needs no dynamic circuits:

- [`phase_propagation_cycle(psi, delta, phi)`][qiu_quantum_computing.phase_propagator.sample_based_manual.phase_propagation_cycle] simulates one cycle for a `PreparableState` `phi` and returns the normalized output and the success probability `P`.
- [`phase_propagate_state(psi_in, deltas, phi)`][qiu_quantum_computing.phase_propagator.sample_based_manual.phase_propagate_state] and [`phase_propagate_state_with_constant_delta(psi_in, delta, num_cycles, phi)`][qiu_quantum_computing.phase_propagator.sample_based_manual.phase_propagate_state_with_constant_delta] simulate one successful cycle per delta.
- [`phase_propagate_state_with_arbitrary_signal(psi_in, signal, max_delta, method)`][qiu_quantum_computing.phase_propagator.sample_based_manual.phase_propagate_state_with_arbitrary_signal] is the counterpart of `QuadraticSignalSampleBasedPhasePropagator`: it decomposes and slices the signal the same way.

The simulation still builds and simulates the preparation circuits of `|phi>`, so the `SynthesisMethod` matters here as well; `DENSE` is the fastest on few qubits.

### Synthesis of the state preparation

The `method` of the propagators and of the simulation defaults to `GATE`, a Qiskit `StatePreparation` synthesized when transpiling. Qiskit's synthesis is unreliable for the nearly uniform `|phi>` of smooth signals (qiskit 2.2): transpiling can fail in its two-qubit decomposition, and it can prepare wrong states (see [`GATE`: Qiskit's `StatePreparation`](#gate-qiskits-statepreparation)). Pass `SynthesisMethod.DECOMPOSED` for the [Möttönen synthesis](#decomposed-uniformly-controlled-rotations), which is numerically robust, or `SynthesisMethod.DENSE` for exact results on few qubits.

### Pitfalls of the phase propagator

- The axis must have `2**n` samples with `n >= 1`, for the direct and the sample-based phases alike.
- The direct phases need a `PolynomialSignal` of power at most 3; arithmetic other than scaling, e.g. `lens + 1`, returns a plain `AlgebraicSignal` without a `power`.
- The sample-based phases need a signal of one sign. A signal of mixed signs can be shifted by a constant, which only changes the global phase, `e^(i (f + c)) = e^(i c) e^(i f)`; the shift increases `|alpha|` and thus the number of cycles, and makes `|phi>` more uniform.
- A vanishing signal raises a `ValueError`, e.g. a sample-based phase scaled by a time of 0.
- The number of cycles grows as `|alpha| / max_delta`, with `alpha` the sum of all samples, i.e. with the number of samples for a signal of fixed magnitude.
