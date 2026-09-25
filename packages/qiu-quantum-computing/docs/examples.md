# Examples

## Comparing the synthesis methods

The three [synthesis methods](../qiu-qiskit-encore/user-guide.md#synthesis-methods) prepare the same state, but transpile into very different circuits. This example prepares a random complex state of 4 qubits with each method, transpiles the circuits into CNOT and single-qubit gates, and compares their CNOT counts.

```python
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

num_qubits = 4
rng = np.random.default_rng(7)
state = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
state /= np.linalg.norm(state)

cnot_counts = {}
for method in SynthesisMethod:
    circuit = state_preparation_circuit(state, method=method)
    transpiled = transpile(circuit, basis_gates=["cx", "u"], seed_transpiler=0)
    # every method prepares the state exactly, global phase included
    assert np.allclose(Statevector(transpiled).data, state)
    cnot_counts[method] = transpiled.count_ops().get("cx", 0)

# the decomposed synthesis stays within its bound of 2**(n+1) - 4 CNOT gates ...
assert cnot_counts[SynthesisMethod.DECOMPOSED] <= 2 ** (num_qubits + 1) - 4
# ... while the generic synthesis of the dense unitary needs many more
assert cnot_counts[SynthesisMethod.DENSE] > 2 * cnot_counts[SynthesisMethod.DECOMPOSED]
```

With qiskit 2.2, Qiskit's synthesis of the `GATE` method needs 11 CNOT gates, the `DECOMPOSED` method 28, and the transpiled `DENSE` unitary 95. `GATE` is the most compact for generic states, `DECOMPOSED` the robust alternative, and `DENSE` best left untranspiled as an exact reference.

## Checking the QFT against `numpy.fft`

The QFT of a periodic state concentrates it on the basis states of its frequency. This example applies the QFT to a cosine of frequency 3 on 5 qubits, checks it against NumPy's orthonormal inverse DFT for every synthesis method, and undoes it with the inverse QFT, which is the forward DFT.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.qft import qft_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

num_qubits = 5
size = 2**num_qubits
j = np.arange(size)
state = np.cos(2 * np.pi * 3 * j / size)
state /= np.linalg.norm(state)

for method in SynthesisMethod:
    transformed = Statevector(state).evolve(qft_circuit(num_qubits, method=method))
    assert np.allclose(transformed.data, np.fft.ifft(state, norm="ortho"))

    # the frequencies 3 and -3, i.e. the basis states |3> and |29>
    probabilities = transformed.probabilities()
    assert np.allclose(probabilities[[3, size - 3]], 0.5)

    # the inverse QFT is the forward DFT, and restores the state
    inverse = qft_circuit(num_qubits, inverse=True, method=method)
    restored = transformed.evolve(inverse)
    assert np.allclose(restored.data, state)
    assert np.allclose(restored.data, np.fft.fft(transformed.data, norm="ortho"))
```

All three methods agree with `numpy.fft.ifft(state, norm="ortho")`, and the state is split evenly between the basis states `|3>` and `|29>`, i.e. the frequencies `3` and `-3` modulo 32.

## Measuring the overlap of two states

A [`PreparableState`][qiu_quantum_computing.preparable_state.PreparableState] provides both the preparation and the un-preparation of a state, which is what a compute-uncompute overlap test needs: preparing `|psi>` and then applying the inverse preparation of `|phi>` leaves `|0...0>` with the probability `|<phi|psi>|**2`. This example builds the circuit, samples it on Aer's CPU simulator with [qiu-qiskit-aer-encore](../qiu-qiskit-aer-encore/index.md), and compares the estimate with the exact overlap.

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

rng = np.random.default_rng(3)
amplitudes = rng.normal(size=(2, 8)) + 1j * rng.normal(size=(2, 8))
psi, phi = (
    PreparableState(data / np.linalg.norm(data), SynthesisMethod.DECOMPOSED)
    for data in amplitudes
)

circuit = QuantumCircuit(psi.num_qubits)
circuit.compose(psi.circuit, inplace=True)
circuit.compose(phi.inverse_circuit, inplace=True)
circuit.measure_all()

shots = 20_000
simulator = aer_simulator(device="cpu")
result = simulator.run(transpile(circuit, simulator), shots=shots, seed_simulator=1)
estimate = result.result().get_counts().get("000", 0) / shots

exact = abs(np.vdot(phi.statevector.data, psi.statevector.data)) ** 2
assert abs(estimate - exact) < 0.02
```

The estimated overlap agrees with the exact `|<phi|psi>|**2` up to the statistical error of the shots, about `0.002` for the overlap of about `0.08` here. Both states cache their circuits, so preparing them again, e.g. in the next iteration of an algorithm, costs nothing.

## Preparing the state of a smooth signal

Algorithms that load a classical signal `f >= 0` into amplitudes prepare the state `sqrt(f / sum(f))`, which is nearly uniform for smooth signals. Qiskit's `StatePreparation` can fail to synthesize such states (see the [User Guide](user-guide.md#gate-qiskits-statepreparation)); the `DECOMPOSED` method prepares them robustly, and without any `rz` gates, since their amplitudes are real and non-negative.

```python
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.state_preparation import state_preparation_circuit
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

num_qubits = 5
x = np.linspace(0, 1, 2**num_qubits)
signal = 1 + 0.01 * x**2
state = np.sqrt(signal / signal.sum())

circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
assert "rz" not in circuit.count_ops()
assert circuit.global_phase == 0

transpiled = transpile(circuit, basis_gates=["cx", "u"])
assert np.allclose(Statevector(transpiled).data, state)
# at most 2**n - 2 CNOT gates for a real non-negative state
assert transpiled.count_ops()["cx"] <= 2**num_qubits - 2
```

The circuit prepares the state exactly with at most `2**n - 2 = 30` CNOT gates, where, with qiskit 2.2, transpiling the `GATE` preparation of this very state raises a `TranspilerError`.

## Building a multiplexed rotation

A uniformly controlled rotation applies a different rotation to its target for each basis state of its controls. This example builds a `ry` multiplexer with 3 controls from random angles, checks it against the block diagonal matrix of the rotations, and counts its CNOT gates.

```python
import numpy as np
import scipy.linalg
from qiskit.quantum_info import Operator
from qiu_quantum_computing.uniformly_controlled_rotation import (
    uniformly_controlled_rotation,
)


def ry(angle: float) -> np.ndarray:
    """The matrix of the rotation about the y axis."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]])


num_controls = 3
angles = np.random.default_rng(0).uniform(-np.pi, np.pi, 2**num_controls)
circuit = uniformly_controlled_rotation("y", angles)

# target qubit 0, controls 1, 2 and 3: the block c rotates the target by angles[c]
assert circuit.num_qubits == num_controls + 1
expected = scipy.linalg.block_diag(*(ry(angle) for angle in angles))
assert np.allclose(Operator(circuit).data, expected)

# 2**k rotations and 2**k CNOT gates
assert circuit.count_ops() == {"ry": 2**num_controls, "cx": 2**num_controls}

# angles independent of the controls need no CNOT gates at all
assert uniformly_controlled_rotation("y", np.full(8, 0.3)).count_ops() == {"ry": 1}
```

The circuit is exactly `diag(ry(angles[0]), ..., ry(angles[7]))` with 8 rotations and 8 CNOT gates, where a naive implementation with multi-controlled rotations would need many more.

## A shifted lens

A lens whose center is displaced to `x0` applies the phase `alpha (x - x0)^2`, which is not a monomial. Expanded, it is the quadratic phase `alpha x^2`, the linear phase `-2 alpha x0 x` and the constant `alpha x0^2`, each applied exactly by a direct phase circuit:

```python
import numpy as np
from qiu_signals.algebraic_signal import PolynomialSignal, QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiu_quantum_computing.phase_propagator.direct import polynomial_phase_circuit

x_axis = PositionAxis(size=16, delta_x=0.1, ordering=IndexOrdering.CENTERED)
alpha, x0 = -3.0, 0.2

monomials = [
    QuadraticSignal(x_axis, alpha),
    PolynomialSignal(x_axis, alpha=-2 * alpha * x0, power=1),
    PolynomialSignal(x_axis, alpha=alpha * x0**2, power=0),
]
lens = QuantumCircuit(4)
for monomial in monomials:
    lens.compose(polynomial_phase_circuit(monomial), inplace=True)

expected = np.exp(1j * alpha * (x_axis.values - x0) ** 2)
assert np.allclose(Operator(lens).data, np.diag(expected))
# 4 + 4 phase gates, 6 controlled phase gates, 2 X gates per CENTERED circuit
assert dict(lens.count_ops()) == {"p": 8, "cp": 6, "x": 4}
```

The composed circuit is the exact diagonal unitary of the shifted lens, global phase included; the constant term is only a global phase of the circuit.

## An arbitrary phase, sample-based

A Gaussian phase `f(x) = 0.1 e^(-x^2)` is no polynomial, so it is applied sample-based. The statevector simulation of `sample_based_manual` runs the cycles one by one; each output is checked against the closed-form cycle map, the total success probability against its lower bound, and the final state against `e^(i f) psi`:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiu_quantum_computing.phase_propagator.sample_based_manual import (
    phase_propagation_cycle,
)

x_axis = PositionAxis(size=16, delta_x=0.25, ordering=IndexOrdering.CENTERED)
f = AlgebraicSignal(x_axis, lambda x: 0.1 * np.exp(-(x**2)))
max_delta = 0.02

alpha, phi = sample_based_decomposition(f)
deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)
weights = np.abs(phi.data) ** 2
assert np.allclose(alpha * weights, f.data)

rng = np.random.default_rng(7)
amplitudes = rng.normal(size=16) + 1j * rng.normal(size=16)
psi_in = Statevector(amplitudes / np.linalg.norm(amplitudes))

psi, closed_form, success = psi_in, psi_in.data, 1.0
prepared_phi = PreparableState(phi, SynthesisMethod.DENSE)
for delta in deltas:
    psi, probability = phase_propagation_cycle(psi, delta, prepared_phi)
    unnormalized = closed_form * (1 + (np.exp(1j * delta) - 1) * weights)
    assert np.isclose(probability, np.vdot(unnormalized, unnormalized).real)
    closed_form = unnormalized / np.linalg.norm(unnormalized)
    success *= probability

assert np.allclose(psi.data, closed_form)
assert success >= 1 - abs(alpha) * max_delta / 4
fidelity = abs(np.vdot(np.exp(1j * f.data) * psi_in.data, psi.data)) ** 2
assert fidelity > 1 - 1e-6
```

The 36 cycles of `delta ~ 0.0196` all succeed with a probability of about 0.9992, above the bound `1 - |alpha| max_delta / 4 ~ 0.9965`, and the output agrees with `e^(i f) psi` to a fidelity of about `1 - 1e-7`.

## The propagator circuit on Aer

`QuadraticSignalSampleBasedPhasePropagator` is the circuit of the same protocol, with measurements and a loop that breaks at the first failure. On the Aer simulator, a successful run agrees with the statevector simulation, which serves as its reference:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)
from qiu_quantum_computing.phase_propagator.sample_based_manual import (
    phase_propagate_state_with_arbitrary_signal,
)

x_axis = PositionAxis(size=4, delta_x=0.5, ordering=IndexOrdering.FFT)
f = AlgebraicSignal(x_axis, lambda x: 0.05 * (1 + np.cos(x)))
max_delta, method = 0.02, SynthesisMethod.DECOMPOSED
psi = Statevector.from_label("+0")

propagator = QuadraticSignalSampleBasedPhasePropagator(f, max_delta, method)
circuit = QuantumCircuit(*propagator.qregs, *propagator.cregs)
circuit.initialize(psi, propagator.qregs[0])
circuit.compose(propagator, inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
result = simulator.run(
    transpile(circuit, simulator, optimization_level=1), shots=1
).result()
assert result.get_counts().int_outcomes().get(0) == 1  # all cycles succeeded
output = np.asarray(result.get_statevector().data)[:4]

simulated = phase_propagate_state_with_arbitrary_signal(psi, f, max_delta, method)
assert propagator.num_of_cycles == 19
assert np.allclose(output, simulated.data)
assert abs(np.vdot(np.exp(1j * f.data) * psi.data, output)) ** 2 > 1 - 1e-6
```

The `DECOMPOSED` preparation avoids Qiskit's unreliable synthesis of the nearly uniform `|phi>` of this smooth signal, and the optimization level 1 keeps the transpiled circuit exact.

## Inspecting every cycle

`GenericIterativeSampleBasedPhasePropagator` takes any preparable `|phi>` and one delta per cycle, which may differ. With `take_snapshot=True`, Aer saves the statevector after each cycle, labeled by its index, so each one can be compared with the closed-form cycle map:

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    GenericIterativeSampleBasedPhasePropagator,
)

phi = PreparableState([0.1, 0.3, 0.5, np.sqrt(0.65)], SynthesisMethod.DENSE)
deltas = [0.1, 0.05, 0.02]
propagator = GenericIterativeSampleBasedPhasePropagator(
    deltas, phi.circuit, phi.inverse_circuit, take_snapshot=True
)

psi = Statevector.from_label("+-")
circuit = QuantumCircuit(*propagator.qregs, *propagator.cregs)
circuit.initialize(psi, propagator.qregs[0])
circuit.compose(propagator, inplace=True)

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
result = simulator.run(
    transpile(circuit, simulator, optimization_level=1), shots=1
).result()
assert result.get_counts().int_outcomes().get(0) == 1

weights = np.abs(phi.statevector.data) ** 2
expected = psi.data
for cycle, delta in enumerate(deltas):
    expected = expected * (1 + (np.exp(1j * delta) - 1) * weights)
    expected = expected / np.linalg.norm(expected)
    snapshot = np.asarray(result.data()[str(cycle)].data)[:4]
    assert np.allclose(snapshot, expected)
```

Each snapshot holds the state of both registers after the reset of `phi`, so its first four amplitudes are the output of `psi` after that cycle, and each agrees with the closed form.
