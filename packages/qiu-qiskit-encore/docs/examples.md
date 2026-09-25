# Examples

## Validating the input of a function

A function taking a state as a Qiskit `Statevector` or as its amplitudes validates it once, and then works with the returned copy. This example computes the probability of each qubit being in `|1>`, from any valid input.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector


def excitation_probabilities(state: Statevector | np.ndarray) -> np.ndarray:
    """Return the probability of each qubit, from qubit 0, to be in |1>."""
    statevector = validated_statevector(state)
    return np.array(
        [
            statevector.probabilities([qubit])[1]
            for qubit in range(statevector.num_qubits)
        ]
    )


# |01> and |11> with equal weights, little-endian: qubit 0 is always excited
state = np.array([0, 1, 0, 1]) / np.sqrt(2)
assert np.allclose(excitation_probabilities(state), [1.0, 0.5])
assert np.allclose(excitation_probabilities(Statevector(state)), [1.0, 0.5])

# an unnormalized state is rejected instead of giving meaningless probabilities
try:
    excitation_probabilities(np.array([0, 1, 0, 1]))
except ValueError as error:
    assert "normalized" in str(error)
else:
    raise AssertionError("expected a ValueError")
```

## Writing a building block taking a synthesis method

A building block of the monorepo takes a `method`, converts raw values with `SynthesisMethod(method)`, and handles every method in its own branch, raising a `NotImplementedError` for the others. This example writes a block applying the Hadamard gate to every qubit, as a dense unitary, as a Qiskit gate or decomposed into single Hadamard gates, and checks that all three are the same unitary.

```python
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate
from qiskit.quantum_info import Operator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod


def hadamard_layer(
    num_qubits: int, method: SynthesisMethod = SynthesisMethod.GATE
) -> QuantumCircuit:
    """Return the circuit applying the Hadamard gate to every qubit."""
    method = SynthesisMethod(method)
    circuit = QuantumCircuit(num_qubits, name="hadamard_layer")
    if method == SynthesisMethod.DENSE:
        matrix = Operator(HGate())
        for _ in range(num_qubits - 1):
            matrix = matrix.tensor(Operator(HGate()))
        circuit.unitary(matrix, range(num_qubits), label="hadamard_layer")
    elif method == SynthesisMethod.GATE:
        layer = QuantumCircuit(num_qubits, name="hadamard_layer")
        layer.h(range(num_qubits))
        circuit.append(layer.to_gate(), range(num_qubits))
    elif method == SynthesisMethod.DECOMPOSED:
        circuit.h(range(num_qubits))
    else:
        raise NotImplementedError(f"The method {method} is not implemented.")
    return circuit


reference = Operator(hadamard_layer(3, SynthesisMethod.DECOMPOSED))
for method in ["dense", "gate", "decomposed"]:  # raw values are accepted
    assert Operator(hadamard_layer(3, method)) == reference
assert np.allclose(np.abs(reference.data), 1 / np.sqrt(8))
```

All three representations implement the same unitary; they differ only in what the circuit contains, and thus in how Qiskit transpiles it.
