# User Guide

`qiskit_pytest_helper` has five modules:

| Module | Contents |
| --- | --- |
| `hypothesis_strategies` | Strategies of quantum states, of axes of `2**n` samples and of moderate coefficients. |
| `assertions` | Exact comparisons of states and operators with Qiskit's equality, and a unitarity check. |
| `circuits` | Typed unitaries and gate counts of circuits, and transpilation without approximation. |
| `propagation` | The Aer simulation of sample-based phase propagators and the closed form they are compared with. |
| `constants` | The bounds of the strategies and the fidelity tolerances of approximate protocols. |

Everything that needs no qubits, i.e. the comparison of plain numbers and the strategies of numbers, axes and signals, is in [python-pytest-helper](../../python-pytest-helper/).

## Quantum state strategies

The strategies of `hypothesis_strategies` generate states of `n` qubits, `MIN_QUBITS <= n <= MAX_QUBITS` (2 to 4), as `2**n` complex amplitudes whose magnitudes lie in `[MIN_MAGNITUDE, MAX_MAGNITUDE]` (0.01 to 1) before any normalization. No amplitude vanishes, so every basis state is populated; all bounds can be passed as arguments, e.g. `min_qubits=1`.

| Strategy | Generates |
| --- | --- |
| [`quantum_state_array`][qiskit_pytest_helper.hypothesis_strategies.quantum_state_array] | Unnormalized NumPy arrays of the amplitudes. |
| [`normalized_quantum_state_array`][qiskit_pytest_helper.hypothesis_strategies.normalized_quantum_state_array] | The same arrays, normalized to 1. |
| [`non_normalized_quantum_state_array`][qiskit_pytest_helper.hypothesis_strategies.non_normalized_quantum_state_array] | Normalized arrays scaled by a complex factor whose norm is not close to 1, e.g. to test the validation of inputs. |
| [`valid_qiskit_statevector`][qiskit_pytest_helper.hypothesis_strategies.valid_qiskit_statevector] | Normalized states as Qiskit `Statevector`s. |
| [`state_pairs_with_equal_qubits`][qiskit_pytest_helper.hypothesis_strategies.state_pairs_with_equal_qubits] | Pairs of `Statevector`s of the same number of qubits, e.g. `psi` and `phi` of a sample-based propagator. |

The strategies are `hypothesis.strategies.composite` functions: they are called to obtain a strategy, `valid_qiskit_statevector()` or `valid_qiskit_statevector(max_qubits=3)`.

## Qubit axes and signals

A signal is represented by `n` qubits if its axis has `2**n` samples. [`qubit_sizes`][qiskit_pytest_helper.hypothesis_strategies.qubit_sizes] generates these sizes, and [`qubit_axes`][qiskit_pytest_helper.hypothesis_strategies.qubit_axes] axes of them in a domain of `python_signals.physical_axis.AxisDomain`:

- Position axes have spacings of at most 1 (between `MIN_MAGNITUDE` and 1), so that the phases of polynomial signals on them stay moderate.
- Fourier axes, e.g. `AxisDomain.MOMENTUM`, are conjugate to such a position axis, momenta with `hbar = 1`, and keep its ordering.
- The orderings are all of `IndexOrdering` by default; circuits acting in the Fourier domain usually need `orderings=st.just(IndexOrdering.FFT)`.

Signals on qubit axes come from the generic signal strategies of `python_pytest_helper.hypothesis_strategies`, which take a strategy of axes, e.g. `positive_polynomial_signals(qubit_axes(AxisDomain.POSITION), total=0.1)` for signals of one sign, as the sample-based protocol needs, or `monomial_signals(qubit_axes(...), alphas=moderate_alphas)`. [`moderate_alphas`][qiskit_pytest_helper.hypothesis_strategies.moderate_alphas] are coefficients between `MIN_MAGNITUDE` and `MAX_MAGNITUDE`, keeping the phases of monomial signals on these axes moderate.

## Assertions with Qiskit's equality

[`assert_equal_states`][qiskit_pytest_helper.assertions.assert_equal_states] and [`assert_equal_operators`][qiskit_pytest_helper.assertions.assert_equal_operators] convert both arguments with `Statevector(...)` and `Operator(...)` and compare them with `==`:

- A state is given as a `Statevector`, its amplitudes, or a circuit, which is compared by the state it prepares from `|0...0>`.
- An operator is given as an `Operator`, its matrix, a gate or a circuit, which is compared by its unitary.
- The dimensions are compared first, with their own message.
- On failure, the message gives the largest deviation of an amplitude or entry; for states also the fidelity, and for operators whether they are equal up to a global phase (`Operator.equiv`).

Qiskit's equality compares the data with `numpy.allclose` at the tolerances of the objects, `rtol=1e-5` and `atol=1e-8` by default. The comparison is exact up to these tolerances, and it includes the global phase:

- A global phase becomes a relative phase as soon as a circuit is controlled or composed into a larger one, and the building blocks of the monorepo are used that way; `RZ(theta)` and `P(theta)` differ by `e^(-i theta / 2)` and are not equal.
- All tests compare at the same standard tolerances instead of tolerances chosen per test, which could be loosened until a wrong result passes.

[`assert_unitary`][qiskit_pytest_helper.assertions.assert_unitary] asserts that an operator, e.g. a matrix, is unitary, with `Operator.is_unitary`.

Plain numbers and arrays that are not states are compared with `python_pytest_helper.assertions.assert_close`, see [python-pytest-helper](../../python-pytest-helper/). The results of approximate protocols, e.g. a phase applied with the sample-based protocol, are not equal to the exact ones; their tests assert a fidelity, `state_fidelity(actual, expected) >= 1 - FIDELITY_TOLERANCE`, see [Constants](#constants).

## Circuit helpers

Qiskit's annotations of `Operator.data` and `QuantumCircuit.count_ops` are too loose or wrong for type checkers, so tests use typed helpers of `circuits` instead:

- [`unitary_matrix(circuit)`][qiskit_pytest_helper.circuits.unitary_matrix]: the dense unitary of a circuit or operator, as a complex NumPy array.
- [`gate_counts(circuit)`][qiskit_pytest_helper.circuits.gate_counts]: the number of gates of each name, as a `dict[str, int]`.

[`transpile_exactly(circuit, backend)`][qiskit_pytest_helper.circuits.transpile_exactly] transpiles a circuit for a backend, e.g. an Aer simulator, at [`EXACT_OPTIMIZATION_LEVEL`][qiskit_pytest_helper.circuits.EXACT_OPTIMIZATION_LEVEL], 1. From optimization level 2, the transpiler removes gates it deems equivalent to the identity, e.g. rotations by angles of `1e-7` or `1e-6`, which changes the amplitudes by as much. The circuits of the monorepo apply many small phases, e.g. the cycles of the sample-based protocol with small `delta`, so a simulation of a circuit transpiled at level 2 or 3 would fail an exact comparison, or silently lose the phases a test is about. Level 1 still maps the circuit to the basis gates of the backend and merges adjacent single-qubit gates, but keeps all of them.

## Simulating sample-based propagators

The sample-based phase propagators of [qiskit-phase-propagator](../../qiskit-phase-propagator/) and [qiskit-hamiltonian-simulation](../../qiskit-hamiltonian-simulation/) act on a register `psi` of `n` qubits, a register `phi` of `n` qubits and `n` classical success flags. Each cycle prepares `|phi>`, applies a partial phase, un-prepares `|phi>` and measures the `phi` register into the flags; the next cycle only runs if the flags are 0.

[`run_propagator(propagator, psi)`][qiskit_pytest_helper.propagation.run_propagator] simulates such a propagator on the initial state `psi`:

1. It initializes the `psi` register to `psi`, composes the propagator and saves the statevector.
2. It transpiles the circuit exactly for the Aer CPU statevector simulator of [qiskit-aer-encore](../../qiskit-aer-encore/), with the seed [`SIMULATOR_SEED`][qiskit_pytest_helper.propagation.SIMULATOR_SEED] (1234) for reproducible measurements, and runs one shot.
3. It returns whether all cycles succeeded, i.e. the flags read 0, and the first `2**n` amplitudes of the final statevector: those of the `psi` register with `phi` in `|0...0>`, exact including their global phase.

Whether the cycles succeed is random, so tests discard failed runs with `hypothesis.assume(succeeded)`. [`exact_cycles(psi, phi, deltas)`][qiskit_pytest_helper.propagation.exact_cycles] is the closed form of successful cycles: each maps the amplitudes `psi_j` to `psi_j (1 + (e^(i delta) - 1) |phi_j|**2)`, renormalized. The output of a successful run is compared with it by `assert_equal_states`, exactly; its distance to `e^(i f) psi` is the error of the protocol, asserted with a fidelity tolerance.

## Constants

`constants` holds the bounds of the quantum strategies and the tolerances of approximate protocols:

| Constant | Value | Meaning |
| --- | --- | --- |
| `MIN_QUBITS` | 2 | The fewest qubits of generated states and qubit axes. |
| `MAX_QUBITS` | 4 | The most qubits of generated states and qubit axes. |
| `MIN_MAGNITUDE` | 0.01 | The smallest magnitude of unnormalized amplitudes, spacings and moderate coefficients. |
| `MAX_MAGNITUDE` | 1.0 | The largest magnitude of unnormalized amplitudes and moderate coefficients. |
| `FIDELITY_TOLERANCE` | 0.0001 | The infidelity an approximate protocol may have, e.g. a sample-based phase. |
| `REDUCED_FIDELITY_TOLERANCE` | 0.01 | The larger infidelity of protocols with larger errors, e.g. the sample-based time evolution. |
| `MAX_NUM_OF_CYCLES` | 200 | The number of cycles a propagator of a test stays below, keeping its simulation fast. |

The `MAX_MAGNITUDE` of this package bounds amplitudes and coefficients; it is not the `MAX_MAGNITUDE` of `python_pytest_helper.hypothesis_strategies` (`1e3`), which bounds generated numbers.

## The Hypothesis profiles

Hypothesis loads its built-in profile `ci` when it runs on CI, i.e. when the environment variable `CI` is set, as on GitHub Actions, and its profile `default` otherwise:

- `ci` has no deadline per example: the first examples of a run in a fresh environment, e.g. importing and compiling Qiskit and Aer, can exceed any deadline.
- It is derandomized and has no example database, so that each run of CI tests the same examples, and it prints how to reproduce a failing example (`print_blob=True`).

To run the tests of a package as CI does, from the repository root:

```sh
CI=true uv run pytest packages/qiskit-phase-propagator
```

With the `default` profile, local runs keep Hypothesis' default deadline of 200 ms per example; tests whose examples simulate circuits set `@settings(deadline=None)` and, where each example is expensive, fewer examples, e.g. `max_examples=10`.
