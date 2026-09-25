# User Guide

`qiskit-encore` has three families of building blocks, each in its own module:

| module                                                                                     | building block                                            |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| [`state_preparation`][qiskit_encore.state_preparation], [`preparable_state`][qiskit_encore.preparable_state] | circuits preparing a state from the all-zero state, and back |
| [`qft`][qiskit_encore.qft]                                                                 | the quantum Fourier transform and its inverse             |
| [`uniformly_controlled_rotation`][qiskit_encore.uniformly_controlled_rotation]             | multiplexed `ry` and `rz` rotations                       |

State preparation and the QFT share the [`SynthesisMethod`][qiskit_encore.synthesis_method.SynthesisMethod] of [`synthesis_method`][qiskit_encore.synthesis_method]; the uniformly controlled rotations are always decomposed into elementary gates.

## Synthesis methods

Every building block can be represented in its circuit in three ways, chosen by a `method` parameter:

| method       | the circuit contains                                                    | use it for                                                                     |
| ------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `DENSE`      | a single unitary gate defined by its dense matrix                       | exact reference results on few qubits (the matrix has `4**n` entries)         |
| `GATE`       | a single high-level Qiskit gate, synthesized by Qiskit when transpiling | the default: leaving the synthesis to Qiskit, e.g. for hardware-aware transpiling |
| `DECOMPOSED` | an already decomposed circuit of elementary gates                       | a synthesis we control, where Qiskit's is not suitable                         |

Some guidelines for the choice:

- `DENSE` is the most direct representation: simulators apply the matrix as it is, which makes it the reference for tests and small simulations. Its memory grows as `4**n`, and transpiling it into elementary gates falls back on Qiskit's generic unitary synthesis, which yields far more CNOT gates than the other methods.
- `GATE` keeps the circuit abstract until it is transpiled, so Qiskit can choose a synthesis for the target, and the circuit stays small and readable. Its quality is the quality of Qiskit's synthesis, including its bugs.
- `DECOMPOSED` gives a circuit of elementary gates of known structure and gate counts. For state preparation, it is synthesized by this package, independently of the Qiskit version, and is the choice wherever Qiskit's synthesis fails; for the QFT, it is Qiskit's standard textbook circuit.

All three methods are exact: they implement the same unitary, up to floating-point rounding, and differ only in their representation. For approximate state preparation with shallower circuits, see [qiskit-mps-initializer](../../qiskit-mps-initializer/).

`SynthesisMethod` is an `ExtendedEnum` of [python-encore](../../python-encore/), whose members compare equal to their raw values. Every function taking a `method` also accepts the raw values `"dense"`, `"gate"` and `"decomposed"`, and converts them to the members:

```python
from qiskit_encore.qft import qft_circuit
from qiskit_encore.synthesis_method import SynthesisMethod

assert SynthesisMethod("dense") is SynthesisMethod.DENSE
assert SynthesisMethod.GATE == "gate"
assert qft_circuit(2, method="dense").count_ops() == {"unitary": 1}
```

A method a function does not implement raises a `NotImplementedError`, so a method added to the enum in the future fails loudly until it is implemented everywhere.

## Conventions

### Qubit ordering

The package follows Qiskit's little-endian convention throughout: the amplitude at index `j = sum_q b_q 2**q` of a statevector belongs to the basis state in which qubit `q` is in `|b_q>`. Qubit 0 is thus the least significant bit of the index. The QFT encodes its integers in the same order, and the controls of a uniformly controlled rotation select its angle by the same rule.

### Global phase

Every circuit of this package implements its operation exactly, *including the global phase*. A state preparation circuit `U` satisfies `U|0...0> = |state>`, not merely `U|0...0> = e^(i phi)|state>`. Results can therefore be compared by their amplitudes, e.g. with `numpy.allclose` or the equality of Qiskit's `Statevector`, rather than only up to a phase with `Statevector.equiv`. The global phase matters as soon as the circuit is controlled, e.g. in phase estimation or in the Hadamard test.

### Input states

The state preparation functions and [`PreparableState`][qiskit_encore.preparable_state.PreparableState] accept a Qiskit `Statevector` or anything NumPy converts to an array of amplitudes. They validate it with [`validated_statevector`][qiskit_encore.state_preparation.validated_statevector], which returns a copy as a complex `Statevector` and raises a `ValueError` unless:

- the dimension is a power of 2 of at least one qubit, i.e. `2, 4, 8, ...`, and
- the state is normalized, as checked by `Statevector.is_valid`, i.e. up to Qiskit's tolerances `Statevector.atol` and `Statevector.rtol`.

States are never normalized silently: normalize them yourself, e.g. with `state / numpy.linalg.norm(state)`.

## State preparation

[`state_preparation_circuit(state, *, method, inverse)`][qiskit_encore.state_preparation.state_preparation_circuit] returns a circuit on `n` qubits for a state of `2**n` amplitudes. With `inverse=False` (the default), it maps `|0...0>` to the state; with `inverse=True`, it maps the state back to `|0...0>`, as needed for un-computing a state or measuring an overlap. The method defaults to `GATE`.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_encore.state_preparation import state_preparation_circuit
from qiskit_encore.synthesis_method import SynthesisMethod

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
    Qiskit's isometry synthesis prepares wrong states, with fidelity 0, when two of its intermediate single-qubit gates are close but not equal. The tests of this package document this with a state of 3 qubits found by `hypothesis`, on which it fails in qiskit 2.2 to 2.5; the test is marked as an expected failure, and alerts once Qiskit fixes it.

!!! warning "Transpiling can fail for nearly uniform states"
    With qiskit 2.2, the synthesis of the `StatePreparation` of a nearly uniform state, such as the state `sqrt(f / sum(f))` of a smooth signal `f`, can fail: Qiskit's decomposition of its uniformly controlled gates produces a single-qubit matrix that it then rejects as not unitary. Transpiling raises a `TranspilerError` (`HighLevelSynthesis is unable to synthesize "state_preparation"`), and building a `Statevector` of the circuit a `ValueError`. Which states are affected is hard to predict. This is why the sample-based circuits of [qiskit-phase-propagator](../../qiskit-phase-propagator/) and [qiskit-hamiltonian-simulation](../../qiskit-hamiltonian-simulation/) are tested with `DECOMPOSED`.

Use `DECOMPOSED` wherever these failures matter, and `GATE` where a hardware-aware synthesis by Qiskit is more important than robustness.

### `DECOMPOSED`: uniformly controlled rotations

[`decomposed_state_preparation(state)`][qiskit_encore.state_preparation.decomposed_state_preparation] implements the synthesis of Möttönen et al., "Transformation of quantum states using uniformly controlled rotations" (2005). It works in two stages:

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

[`state_preparation_unitary(state)`][qiskit_encore.state_preparation.state_preparation_unitary] returns a Qiskit `Operator` whose first column is the state: `-e^(i theta) H`, where `H` is the Householder reflection swapping `|0...0>` and `-e^(-i theta)|state>`, and `theta` is the phase of the first amplitude. The reflection vector has norm at least 1, so the construction is numerically stable for every state, and it is exact. The `DENSE` circuit holds this operator as a single `unitary` gate labeled `state_preparation`, or its adjoint labeled `state_preparation_dg` for `inverse=True`. The matrix has `4**n` entries, so it is meant for few qubits.

The other columns of the unitary are fixed by the reflection and carry no meaning: only `U|0...0>` is specified.

## Preparable states

[`PreparableState(statevector, method)`][qiskit_encore.preparable_state.PreparableState] bundles a state with its synthesis method, for code that prepares and un-prepares the same state repeatedly, e.g. in every cycle of a protocol:

- `statevector` is the validated copy of the given state, and `method` the synthesis method, `GATE` by default. Raw method values are converted, as everywhere.
- `num_qubits` is the number of qubits of the state.
- `circuit` and `inverse_circuit` are the circuits of `state_preparation_circuit` with `inverse=False` and `inverse=True`. They are built on first access and cached.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState

phi = PreparableState([0.6, 0.8j], method="decomposed")
assert phi.circuit is phi.circuit  # built once
assert np.allclose(Statevector(phi.circuit).data, [0.6, 0.8j])
assert np.allclose(phi.statevector.evolve(phi.inverse_circuit).data, [1, 0])
```

A `PreparableState` is a frozen dataclass: its fields cannot be reassigned, and it holds a copy of the amplitudes, so changing the given array later does not affect it. The cached circuits therefore always prepare the state. Two preparable states are equal when their methods are equal and their statevectors are equal by Qiskit's `Statevector` equality, i.e. up to Qiskit's tolerances. Since a Qiskit `Statevector` is not hashable, neither is a `PreparableState`: it cannot be used as a dictionary key or in a set.

The cached circuits are ordinary, mutable `QuantumCircuit`s. Compose them into other circuits rather than modifying them in place, or the cache no longer matches the state.

## Quantum Fourier transform

[`qft_circuit(num_qubits, *, inverse, method)`][qiskit_encore.qft.qft_circuit] returns the QFT on `num_qubits` qubits, at least 1, or its inverse with `inverse=True`:

| method           | the circuit contains                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| `GATE` (default) | a single Qiskit `QFTGate`, or its inverse, synthesized when transpiling       |
| `DECOMPOSED`     | Hadamard, controlled phase and swap gates of Qiskit's `synth_qft_full`         |
| `DENSE`          | a single unitary gate of the matrix of [`qft_matrix`][qiskit_encore.qft.qft_matrix], or its adjoint |

The QFT maps the basis state `|j>` to `sum_k e^(2 pi i j k / N) |k> / sqrt(N)` with `N = 2**n`, where the integers are encoded in little-endian order like the statevector indices. [`qft_matrix(num_qubits)`][qiskit_encore.qft.qft_matrix] returns this matrix, with the entries `e^(2 pi i j k / N) / sqrt(N)`; it is the matrix of Qiskit's `QFTGate`.

!!! note "The NumPy convention"
    The sign of the exponent is positive, so on the amplitudes the QFT is NumPy's orthonormal **inverse** discrete Fourier transform, and the inverse QFT is the forward one:

    | circuit                           | on the amplitudes `psi`             |
    | --------------------------------- | ----------------------------------- |
    | `qft_circuit(n)`                  | `numpy.fft.ifft(psi, norm="ortho")` |
    | `qft_circuit(n, inverse=True)`    | `numpy.fft.fft(psi, norm="ortho")`  |

    The unnormalized transforms of NumPy's default `norm="backward"` differ from these by the factors `sqrt(N)` and `1 / sqrt(N)`.

The swap gates at the end of `synth_qft_full` are part of the transform, so all three methods implement the same unitary, without a reversal of the qubits left to the user.

!!! note "Transpiling the swaps away"
    From optimization level 2, the default of `transpile`, Qiskit may elide the final swaps of the QFT into a relabeling of the qubits, the `final_layout` of the transpiled circuit. Measurements account for it, but a statevector saved by a simulator, e.g. with Aer's `save_statevector`, then has its qubits permuted. Transpile with `optimization_level=1` where the statevector matters, see [qiskit-aer-encore](../../qiskit-aer-encore/user-guide/#exact-statevectors-of-transpiled-circuits).

## Uniformly controlled rotations

[`uniformly_controlled_rotation(axis, angles)`][qiskit_encore.uniformly_controlled_rotation.uniformly_controlled_rotation] returns a multiplexed rotation, the building block of the `DECOMPOSED` state preparation. For `2**k` angles, the circuit acts on `k + 1` qubits:

- qubit 0 is the target, and qubits `1, ..., k` are the controls;
- for the controls in the basis state `|c>`, with `c = sum_j c_j 2**j` over the control qubits `j + 1`, the target is rotated by `R_axis(angles[c])`.

The circuit thus implements the block diagonal unitary `diag(R(angles[0]), R(angles[1]), ...)`, with `R` being `ry` or `rz` for the [`RotationAxis`][qiskit_encore.uniformly_controlled_rotation.RotationAxis] `"y"` or `"z"`.

```python
import numpy as np
from qiskit.quantum_info import Operator
from qiskit_encore.uniformly_controlled_rotation import uniformly_controlled_rotation

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

## Summary of the pitfalls

- `GATE` state preparation inherits Qiskit's synthesis: wrong states for some inputs, and failing transpilation of nearly uniform states in qiskit 2.2. Use `DECOMPOSED` where this matters.
- `DENSE` circuits are exact but grow as `4**n`, and transpile into many CNOT gates. Keep them for references on few qubits.
- The QFT is NumPy's orthonormal *inverse* DFT, not the forward one.
- From optimization level 2, transpiling may elide the swaps of the QFT into a permutation of the qubits, which statevectors saved by a simulator do not undo; use `optimization_level=1` for exact statevectors.
- The circuits include the global phase; do not drop it when composing controlled versions.
- States must be normalized and of a power-of-2 dimension; they are not normalized for you.
- Do not modify the cached circuits of a `PreparableState` in place.
