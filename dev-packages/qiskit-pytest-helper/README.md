# Qiskit Pytest Helper

Shared helpers for the unit tests of this monorepo, not meant to be published.

- `hypothesis_strategies`: `hypothesis` strategies for quantum states (`valid_qiskit_statevector`, `state_pairs_with_equal_qubits`, ...), for `python-signals` axes of `2**n` samples (`position_axis`, `momentum_axis`, `qubit_axis`, with an optional `forced_ordering`) and for signals on them (`random_positive_signal`, `random_polynomial_signal`).
- `assertions`: `assert_equal_states`, `assert_equal_operators` and `assert_unitary` compare states, operators, matrices and circuits exactly, global phase included, with Qiskit's `Statevector` and `Operator` equality and its default tolerances, instead of tolerances chosen per test.
- `circuits`: typed `unitary_matrix` and `gate_counts` of circuits, working around Qiskit's loose type annotations, and `transpile_exactly`, transpiling at the highest optimization level (1) that does not drop gates close to the identity.
- `propagation`: `run_propagator` simulates a sample-based phase propagator on Aer and returns whether it succeeded and the exact final amplitudes; `exact_cycles` is the closed form they are compared with.
- `constants`: tolerances and bounds shared by the tests.
