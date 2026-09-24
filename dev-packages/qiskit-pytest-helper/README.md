# Qiskit Pytest Helper

Shared quantum helpers for the unit tests of this monorepo, not meant to be published. It builds on [`python-pytest-helper`](../python-pytest-helper/README.md), which holds the floating-point comparisons and the strategies of numbers, axes and signals; this package only adds what involves qubits or Qiskit.

- `hypothesis_strategies`: strategies for quantum states (`valid_qiskit_statevector`, `state_pairs_with_equal_qubits`, ...) and for axes of `2**n` samples representable by `n` qubits (`qubit_sizes`, `qubit_axes`), with `moderate_alphas` for coefficients keeping the phases of monomial signals moderate. Signals on qubit axes come from the generic strategies, e.g. `positive_polynomial_signals(qubit_axes(AxisDomain.POSITION), total=0.1)`.
- `assertions`: `assert_equal_states`, `assert_equal_operators` and `assert_unitary` compare states, operators, matrices and circuits exactly, global phase included, with Qiskit's `Statevector` and `Operator` equality and its default tolerances, instead of tolerances chosen per test. Plain numbers are compared with `python_pytest_helper.assertions.assert_close`.
- `circuits`: typed `unitary_matrix` and `gate_counts` of circuits, working around Qiskit's loose type annotations, and `transpile_exactly`, transpiling at the highest optimization level (1) that does not drop gates close to the identity.
- `propagation`: `run_propagator` simulates a sample-based phase propagator on Aer and returns whether it succeeded and the exact final amplitudes; `exact_cycles` is the closed form they are compared with.
- `constants`: the bounds of the quantum strategies and the fidelity tolerances of approximate protocols.
