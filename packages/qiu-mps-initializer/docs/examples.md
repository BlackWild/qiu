# Examples

## Fidelity of increasing numbers of layers

How many layers does a state need? This example approximates a random complex state of 5 qubits with 1, 2, 4 and 8 layers, and computes the fidelity of each preparation, checking the reported `error` along the way.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_mps_initializer.state_preparation import mps_state_preparation

rng = np.random.default_rng(0)
state = rng.normal(size=32) + 1j * rng.normal(size=32)
state /= np.linalg.norm(state)

fidelities = {}
for max_layers in (1, 2, 4, 8):
    preparation = mps_state_preparation(state, max_layers=max_layers)
    prepared = Statevector(preparation.circuit).data

    # a random state is not reached within so few layers
    assert preparation.num_layers == max_layers and not preparation.converged
    # the error is the distance to the target, global phase included
    assert np.isclose(preparation.error, np.linalg.norm(prepared - state))
    # ... and bounds the fidelity from below
    fidelity = abs(np.vdot(state, prepared)) ** 2
    assert fidelity >= (1 - preparation.error**2 / 2) ** 2 - 1e-12
    fidelities[max_layers] = fidelity

assert fidelities[1] < 0.8 < 0.99 < fidelities[8]
assert fidelities[1] < fidelities[2] < fidelities[4] < fidelities[8]
```

The fidelity grows from about `0.73` with one layer to above `0.99` with 8 layers. For this state it increases steadily up to 8 layers, but not beyond: 16 layers give a lower fidelity than 8, since the greedy construction does not decrease the error monotonically.

## Shallow preparation of a smooth state

Smooth functions sampled on a grid are weakly entangled, which is where MPS layers pay off. This example prepares a Gaussian on 6 qubits with a tolerance on the error, and compares the CNOT count after transpiling with the exact `DECOMPOSED` preparation of [qiu-quantum-computing](../qiu-quantum-computing/index.md).

```python
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_mps_initializer.state_preparation import mps_state_preparation

x = np.linspace(-1, 1, 64)
state = np.exp(-(x**2) / 0.1)
state /= np.linalg.norm(state)

preparation = mps_state_preparation(state, max_layers=10, tolerance=0.05)
assert preparation.converged and preparation.num_layers == 1
fidelity = abs(np.vdot(state, Statevector(preparation.circuit).data)) ** 2
assert fidelity > 0.998


def cnot_count(circuit) -> int:
    """The number of CNOT gates of the circuit transpiled to CNOT and U gates."""
    transpiled = transpile(circuit, basis_gates=["cx", "u"], seed_transpiler=0)
    return transpiled.count_ops()["cx"]


exact = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
assert cnot_count(preparation.circuit) <= 3 * (6 - 1)
assert cnot_count(exact) > 3 * cnot_count(preparation.circuit)
```

A single layer reaches the error `0.05`, a fidelity above `0.998`, with at most 15 CNOT gates, where the exact preparation needs 62. Tighter tolerances need more layers: for this state, `tolerance=0.01` takes about 30 layers, more CNOT gates than the exact preparation, so MPS layers are the tool for moderate accuracies.

## Preparing the state of an intensity signal

An intensity profile `f >= 0` is loaded into a register as the state `|psi> = sqrt(f / alpha)` with `alpha = sum(f)`, so that `f = alpha |psi|**2`. This example samples a Gaussian intensity on a centered position axis of [qiu-signals](../qiu-signals/index.md), splits it with `sample_based_decomposition` of [qiu-quantum-computing](../qiu-quantum-computing/index.md), and reconstructs the intensity from the approximately prepared state.

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

axis = PositionAxis(size=32, delta_x=0.1, ordering=IndexOrdering.CENTERED)
intensity = Signal(axis, np.exp(-(axis.values**2) / 0.5))
alpha, state = sample_based_decomposition(intensity)
assert np.isclose(alpha, intensity.data.sum())

preparation = mps_state_preparation(state, max_layers=10, tolerance=0.02)
assert preparation.converged and preparation.num_layers == 2

reconstructed = alpha * np.abs(Statevector(preparation.circuit).data) ** 2
assert np.max(np.abs(reconstructed - intensity.data)) < 0.05 * intensity.data.max()
```

Two layers reach the tolerance, and the reconstructed intensity deviates from the sampled one by less than 5 % of its peak. The global phase of the prepared state does not matter here, since only `|psi|**2` enters the intensity.

## Inspecting the layer of a bond-2 state

GHZ and W states have bond dimension 2 for any number of qubits, so one layer prepares them exactly. This example builds the bond-2 MPS of the W state of 6 qubits with `quimb`, completes its tensors to unitaries, places them into a layer, and checks that the full preparation needs just this one layer.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_mps_initializer.mps import (
    bond2_mps_approximation,
    disentangler_matrices,
    mps_layer,
)
from qiu_mps_initializer.state_preparation import mps_state_preparation

num_qubits = 6
w_state = np.zeros(2**num_qubits)
w_state[[2**qubit for qubit in range(num_qubits)]] = num_qubits**-0.5

mps = bond2_mps_approximation(w_state)
assert mps.num_tensors == num_qubits and mps.max_bond() == 2

# one 4 x 4 unitary per site but the last, and a 2 x 2 one for the last site
matrices = disentangler_matrices(mps)
assert [matrix.shape for matrix in matrices] == [(4, 4)] * 5 + [(2, 2)]
for matrix in matrices:
    assert np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix)))

layer = mps_layer(bond2_mps_approximation(w_state))
assert np.allclose(Statevector(layer).data, w_state)

preparation = mps_state_preparation(w_state, max_layers=5)
assert preparation.num_layers == 1 and preparation.converged
assert preparation.error < 1e-12
```

The MPS of the W state is exact at bond dimension 2, its 6 unitaries form one layer preparing the state exactly, and `mps_state_preparation` stops after this layer, well within its budget of 5.
