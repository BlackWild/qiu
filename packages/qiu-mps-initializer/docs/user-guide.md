# User Guide

The package has two modules:

| module                                                        | contents                                                                                     |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [`mps`][qiu_mps_initializer.mps]                           | the MPS of bond dimension 2 of a state, and the layer of gates preparing it exactly           |
| [`state_preparation`][qiu_mps_initializer.state_preparation] | the approximate preparation of any state with several such layers, and its error            |

Most users only need [`mps_state_preparation`][qiu_mps_initializer.state_preparation.mps_state_preparation]; the functions of `mps` are its building blocks, public for inspection and for custom constructions.

## The idea

Exact state preparation of a generic state of `n` qubits needs of the order of `2**n` CNOT gates. Many states of interest, e.g. smooth functions sampled on a grid, have little entanglement between the qubits, and are well approximated by a matrix product state (MPS) of small bond dimension. An MPS of bond dimension 2 is prepared *exactly* by a single layer of `n - 1` two-qubit gates on neighboring qubits and one single-qubit gate, i.e. at most `3 (n - 1)` CNOT gates after transpiling. Stacking several such layers approximates states of higher bond dimension better and better.

All preparations of this package are exact for the states they target, i.e. the MPS of bond dimension 2; the approximation lies only in the truncation of a state to such an MPS.

## Matrix product states of bond dimension 2

[`bond2_mps_approximation(state)`][qiu_mps_initializer.mps.bond2_mps_approximation] truncates a state of 2 qubits or more to a `quimb` `MatrixProductState` of bond dimension at most 2:

- The state is validated by `validated_statevector` of [qiu-qiskit-encore](../../qiu-qiskit-encore/): a Qiskit `Statevector` or its amplitudes, normalized, of a power-of-2 dimension. Single-qubit states raise a `ValueError`, since they have no MPS of 2 sites or more.
- The MPS is built by successive singular value decompositions, keeping the 2 largest singular values at each bond. This is the standard truncation, not necessarily the closest MPS of bond dimension 2 to the state.
- It is normalized, right-canonical, and has its tensor indices in the order left, physical, right (`"lpr"`), as the next step expects.

States of up to 3 qubits and product states have bond dimension at most 2, so their MPS is exact.

!!! note "Site and qubit ordering"
    Site `i` of the MPS is the qubit `n - 1 - i`: the MPS lists the qubits from the most significant one down, in the order in which the amplitudes of a Qiskit statevector are indexed, e.g. `|q_(n-1) ... q_1 q_0>`. Qubit 0, the least significant, is the last site.

## The layer of an MPS

[`disentangler_matrices(mps)`][qiu_mps_initializer.mps.disentangler_matrices] completes the tensors of the MPS to unitary matrices:

- one `4 x 4` unitary per site but the last, acting on the qubits of the site and of the next site, and
- one `2 x 2` unitary for the last site.

The tensors of a right-canonical MPS are isometries, which form the first one or two columns of these unitaries; the remaining columns are completed by an orthonormal basis of their null space. Bonds of dimension 1, e.g. of product states, are first expanded to dimension 2, so that all tensors have the same shapes, and tensors whose second column then vanishes are completed as well.

[`mps_layer(mps)`][qiu_mps_initializer.mps.mps_layer] places these unitaries into a circuit `U` on one qubit per site, with `U|0...0>` the state of the MPS:

1. the two-qubit unitaries `G0, G1, ..., G(n-2)` on the neighboring qubits `(n-2, n-1)`, `(n-3, n-2)`, ..., `(0, 1)`, from the most significant pair down,
2. then the single-qubit unitary `G(n-1)` on qubit 0.

The circuit holds them as `unitary` gates labeled `G0`, `G1`, ...; transpiling decomposes each two-qubit unitary into at most 3 CNOT gates.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_mps_initializer.mps import bond2_mps_approximation, mps_layer

ghz = np.zeros(16)
ghz[[0, 15]] = 2**-0.5  # (|0000> + |1111>) / sqrt(2), of bond dimension 2

layer = mps_layer(bond2_mps_approximation(ghz))
assert layer.count_ops() == {"unitary": 4}
assert np.allclose(Statevector(layer).data, ghz)
```

!!! warning "Requirements on the MPS"
    `disentangler_matrices` and `mps_layer` expect a right-canonical MPS with indices in the order left, physical, right, of 2 sites or more and bond dimension at most 2, e.g. of `bond2_mps_approximation`. They check the number of sites and the bond dimension, raising a `ValueError`, but not the canonical form or the index order: an MPS built otherwise with `quimb` must be brought into this form first, e.g. with `right_canonicalize` and `permute_arrays(shape="lpr")`. They also expand the bonds of the given MPS to dimension 2 *in place*.

## Multi-layer state preparation

[`mps_state_preparation(state, max_layers, tolerance=None)`][qiu_mps_initializer.state_preparation.mps_state_preparation] prepares any state of at least one qubit with at most `max_layers` layers, greedily:

1. The residual starts as the target state, `|r_0> = |state>`.
2. Layer `k + 1` is the `mps_layer` of the bond-2 approximation of the residual `|r_k>`, and the residual is updated to `|r_(k+1)> = U_(k+1)^dagger |r_k>`, i.e. `|r_k> = U_k^dagger ... U_1^dagger |state>`.
3. This repeats until the residual reaches `|0...0>`, or `max_layers` layers are built.

If the residual is `|0...0>`, then `U_1 U_2 ... U_k |0...0> = |state>`: the circuit applies the *last* layer found first, and the first layer found last. Each layer disentangles part of what the layers before it leave, so the residual approaches `|0...0>`.

The result is an immutable [`MPSStatePreparation`][qiu_mps_initializer.state_preparation.MPSStatePreparation]:

| attribute    | meaning                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------- |
| `circuit`    | the circuit preparing the state approximately from the all-zero state, its layers applied in order |
| `layers`     | the layers of the circuit, as a tuple in the order they are applied                       |
| `num_layers` | the number of layers                                                                      |
| `error`      | the Euclidean distance of the prepared state to the target, global phase included         |
| `converged`  | whether the prepared state reached the target within the tolerance                        |

`max_layers` is required and must be at least 1; the circuit never has more layers. The construction is deterministic: a run with fewer layers consists of the first layers found by a run with more, i.e. of the last entries of its `layers`.

### When the target is reached

The residual has reached `|0...0>`:

- by default, `tolerance=None`, when it equals `|0...0>` as a Qiskit `Statevector`, i.e. up to Qiskit's tolerances `Statevector.atol` and `Statevector.rtol`: the preparation is exact up to rounding;
- with a `tolerance`, when the distance of the residual to `|0...0>`, which is the `error`, is below it.

No layer is added once the target is reached, so the all-zero state needs no layer at all, and the circuit is empty. If `max_layers` is reached first, `converged` is `False`, and the circuit consists of all `max_layers` layers; the `error` still reports its quality.

### Error and fidelity

The `error` is the Euclidean distance of the prepared state to the target, *including the global phase*, so it is 0 only if the circuit prepares the state exactly, phase included. It bounds the fidelity `F = |<state|U|0...0>|**2` from below: since `error**2 = 2 - 2 Re <state|U|0...0>`,

```text
F >= (1 - error**2 / 2)**2    for error**2 <= 2
```

A small error thus guarantees a high fidelity; a large error may still come with a high fidelity if it is due to a global phase.

## Exact cases

One layer prepares exactly, up to rounding:

- every state of up to 3 qubits, since its bond dimension is at most 2,
- single-qubit states, prepared by one single-qubit unitary, the Householder reflection `state_preparation_unitary` of qiu-quantum-computing,
- product states, of bond dimension 1,
- and all other states of bond dimension at most 2, e.g. GHZ and W states of any number of qubits.

For these states, `mps_state_preparation` returns one layer with `converged` set, whatever `max_layers` is; the all-zero state itself needs no layer.

## Convergence and cost

For states of 4 qubits or more with bond dimension above 2, the error decreases with the number of layers, though *not always monotonically*: a run with more layers can end with a larger error than a run with fewer, e.g. 16 layers worse than 8 for a random state of 5 qubits. For highly entangled states, e.g. random ones, the error decreases slowly, and reaching a small tolerance can take more layers, and thus more CNOT gates, than exact state preparation. The method pays off for weakly entangled states, e.g. smooth signals, where one or a few layers already give small errors.

Some practical guidelines:

- Choose `max_layers` as the budget of circuit depth, and `tolerance` as the accuracy needed; the result reports which one stopped the construction.
- Since the error is not monotonic, scan the number of layers where the best circuit within a budget matters, and keep the one with the smallest `error`.
- Each layer costs `n - 1` two-qubit gates, i.e. at most `3 (n - 1)` CNOT gates, and has depth linear in `n`, since its gates act one after another along the chain of qubits.
- The construction updates the residual as a dense statevector with the dense operator of each layer, which takes `O(4**n)` memory: it is meant for the numbers of qubits of dense statevector simulations, not for large MPS.

## Intensity signals

To load a real signal `f` of one sign, e.g. an intensity profile of [qiu-signals](../../qiu-signals/), into the amplitudes of a register, split it into its sum `alpha` and the state `|psi> = sqrt(f / alpha)`, such that `f = alpha |psi|**2`, with `sample_based_decomposition` of [qiu-quantum-computing](../../qiu-quantum-computing/), and prepare `|psi>`:

```python
import numpy as np
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiu_signals.signal import Signal
from qiskit.quantum_info import Statevector
from qiu_mps_initializer.state_preparation import mps_state_preparation
from qiu_quantum_computing.phase_propagator.sample_based import (
    sample_based_decomposition,
)

intensity = Signal(PositionAxis(8, 1.0, IndexOrdering.NATURAL), np.arange(1.0, 9.0))
alpha, state = sample_based_decomposition(intensity)

preparation = mps_state_preparation(state, max_layers=1)  # 3 qubits: exact
prepared = Statevector(preparation.circuit)
assert np.allclose(alpha * np.abs(prepared.data) ** 2, intensity.data)
```

## Changes from version 0.2

The functionality of version 0.2 is carried over onto the packages of this monorepo:

| 0.2                                                                | now                                                                                           |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `QuantumState.from_dense_data(data, normalize)`                    | a Qiskit `Statevector`, validated by `qiu_qiskit_encore`                                     |
| `QuantumState.generate_mps_initializer_circuit(layers)`            | `mps_state_preparation(state, max_layers).circuit`                                            |
| `QuantumIntensity`                                                 | `qiu_quantum_computing.phase_propagator.sample_based.sample_based_decomposition`                             |
| `helpers.mps_technique.G_matrices`                                 | [`mps.disentangler_matrices`][qiu_mps_initializer.mps.disentangler_matrices]               |
| `helpers.mps_technique.multi_layered_circuit_for_non_approximated` | [`mps_state_preparation`][qiu_mps_initializer.state_preparation.mps_state_preparation]     |
| `helpers.sampling_and_data_preperation`                            | `qiu_signals` axes and `AlgebraicSignal.from_sympy`                                        |
| `utils.simulate_statevector`, `simulate_quantum_info`              | `Statevector(circuit)`, or `qiu_qiskit_aer_encore.simulator.aer_simulator`                        |

Besides:

- `max_layers` is required and bounds the number of layers; formerly, one layer more than asked for could be added, and without a maximum the construction could run forever.
- Single-qubit states are supported.
- States are no longer normalized for you: pass normalized states, e.g. `data / numpy.linalg.norm(data)`.
- The package no longer depends on `pydantic`, `pydantic-numpy` and `qiskit-aer`; simulate the circuits with `Statevector(circuit)` or with [qiu-qiskit-aer-encore](../../qiu-qiskit-aer-encore/).
