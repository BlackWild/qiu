# qiskit-pytest-helper

Shared quantum helpers for the unit tests of this monorepo, not meant to be published. It provides Hypothesis strategies of quantum states and of axes representable by qubits, assertions comparing states and operators with Qiskit's equality, typed circuit helpers, and the Aer simulation of sample-based phase propagators.

It builds on [python-pytest-helper](../python-pytest-helper/), which holds the floating-point comparisons and the strategies of numbers, axes and signals; this package only adds what involves qubits or Qiskit. The tests of the quantum packages, e.g. [qiskit-encore](../qiskit-encore/), [qiskit-phase-propagator](../qiskit-phase-propagator/) and [qiskit-hamiltonian-simulation](../qiskit-hamiltonian-simulation/), use it, so that they share the same strategies, bounds and tolerances.

## Installation

The package is a member of the uv workspace and is installed with all others, from the repository root:

```sh
uv sync --all-packages
```

The `test` dependency group of the root `pyproject.toml` includes it. A package whose tests use it declares it in its own `test` dependency group, together with `python-pytest-helper`, and takes both from the workspace:

```toml
[dependency-groups]
test = [
    "hypothesis>=6.140.3",
    "pytest>=9.0.0",
    "python-pytest-helper",
    "qiskit-pytest-helper",
]

[tool.uv.sources]
python-pytest-helper = { workspace = true }
qiskit-pytest-helper = { workspace = true }
```

The workflow publishing a package tests it with only the dependencies it declares (`uv sync --package <name> --group test`), so a missing declaration fails there. Declared in a dependency group, it is not a dependency of the published package.

## Quick start

A property-based test of Qiskit's state preparation: every generated state is prepared exactly, global phase included.

```python
from hypothesis import given
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


@given(state=valid_qiskit_statevector())
def test_prepares_the_state(state: Statevector):
    """Test that the circuit prepares the state from |0...0>."""
    circuit = QuantumCircuit(state.num_qubits)
    circuit.prepare_state(state)
    assert_equal_states(circuit, state)


test_prepares_the_state()
```

## Where next

- The [User Guide](user-guide.md) explains the strategies, the assertions and why they compare as they do, the circuit and propagator helpers, the constants and the Hypothesis profiles on CI and locally.
- The [Examples](examples.md) are complete tests of state preparations, operators, transpiled circuits and sample-based propagators.
- The [API Reference](reference/qiskit_pytest_helper/index.md) documents every module.
