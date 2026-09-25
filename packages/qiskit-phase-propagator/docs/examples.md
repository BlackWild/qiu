# Examples

## A shifted lens

A lens whose center is displaced to `x0` applies the phase `alpha (x - x0)^2`, which is not a monomial. Expanded, it is the quadratic phase `alpha x^2`, the linear phase `-2 alpha x0 x` and the constant `alpha x0^2`, each applied exactly by a direct phase circuit:

```python
import numpy as np
from python_signals.algebraic_signal import PolynomialSignal, QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_phase_propagator.direct import polynomial_phase_circuit

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
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.sample_based import (
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiskit_phase_propagator.sample_based_manual import phase_propagation_cycle

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
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer_encore.simulator import aer_simulator
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)
from qiskit_phase_propagator.sample_based_manual import (
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
from qiskit_aer_encore.simulator import aer_simulator
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.sample_based import (
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
