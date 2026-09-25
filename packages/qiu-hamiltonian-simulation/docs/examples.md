# Examples

## Free evolution of a Gaussian wave packet

A free particle of mass `m` has the kinetic energy `T(p) = p^2 / (2 m)`, so its evolution is exact with the direct momentum-domain circuit. A Gaussian wave packet of width `sigma` and mean momentum `p0`, starting at `x0`, moves to `x0 + p0 t / m` and spreads to the width `sigma sqrt(1 + (hbar t / (2 m sigma^2))^2)`:

```python
import numpy as np
from qiu_signals.algebraic_signal import QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit.quantum_info import Statevector
from qiu_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)

hbar, mass, t = 1.0, 1.0, 1.5
x_axis = PositionAxis(size=64, delta_x=0.25, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)

x = x_axis.values
x0, p0, sigma = -2.0, 2.0, 1.0
packet = np.exp(-((x - x0) ** 2) / (4 * sigma**2) + 1j * p0 * x / hbar)
psi = Statevector(packet / np.linalg.norm(packet))

kinetic = QuadraticSignal(p_axis, alpha=1 / (2 * mass))
out = psi.evolve(MomentumDomainEvolutionQuadratic((-t / hbar) * kinetic))

# the same evolution with NumPy's FFT
expected = np.fft.ifft(
    np.exp(-1j * t / hbar * kinetic.data) * np.fft.fft(psi.data, norm="ortho"),
    norm="ortho",
)
assert np.allclose(out.data, expected)

# the analytic motion and spreading of the packet
probability = np.abs(out.data) ** 2
mean = np.sum(x * probability)
width = np.sqrt(np.sum((x - mean) ** 2 * probability))
assert abs(mean - (x0 + p0 * t / mass)) < 1e-5
assert abs(width - sigma * np.sqrt(1 + (hbar * t / (2 * mass * sigma**2)) ** 2)) < 1e-5
```

The circuit agrees with NumPy's FFT to rounding, and the packet on 6 qubits follows the analytic mean position `1.0` and width `1.25` to within `1e-6`, since it stays far from the edges of the periodic window.

## A harmonic oscillator by operator splitting

For `H = p^2 / (2 m) + m omega^2 x^2 / 2`, both terms are quadratic, so each is exact with a direct circuit, but they do not commute. The second-order (Strang) splitting approximates the evolution over half a period, after which a displaced packet has moved to the mirrored position; the reference is the matrix exponential of the discretized Hamiltonian:

```python
import numpy as np
from qiu_signals.algebraic_signal import QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiu_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
    PositionDomainEvolutionQuadratic,
)
from scipy.linalg import expm

hbar, mass, omega = 1.0, 1.0, 1.0
num_qubits = 5
x_axis = PositionAxis(size=2**num_qubits, delta_x=0.4, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)
V = QuadraticSignal(x_axis, alpha=mass * omega**2 / 2)
T = QuadraticSignal(p_axis, alpha=1 / (2 * mass))

t, steps = np.pi / omega, 20  # half a period
dt = t / steps
step = QuantumCircuit(num_qubits)
step.compose(PositionDomainEvolutionQuadratic((-dt / (2 * hbar)) * V), inplace=True)
step.compose(MomentumDomainEvolutionQuadratic((-dt / hbar) * T), inplace=True)
step.compose(PositionDomainEvolutionQuadratic((-dt / (2 * hbar)) * V), inplace=True)

x = x_axis.values
packet = np.exp(-((x - 2.0) ** 2) / 2)  # the ground state, displaced to x = 2
psi_in = Statevector(packet / np.linalg.norm(packet))
psi = psi_in
for _ in range(steps):
    psi = psi.evolve(step)

# H = V + F^-1 T F, with F the orthonormal DFT
F = np.fft.fft(np.eye(2**num_qubits), norm="ortho", axis=0)
H = np.diag(V.data) + F.conj().T @ np.diag(T.data) @ F
exact = expm(-1j * t / hbar * H) @ psi_in.data

assert abs(np.vdot(exact, psi.data)) ** 2 > 1 - 1e-4
assert abs(np.sum(x * np.abs(psi.data) ** 2) - (-2.0)) < 1e-3
```

With 20 steps, the splitting agrees with the exact evolution of the discretized Hamiltonian to an infidelity of about `2e-5`, and the packet arrives at `x = -2`.

## A potential step, sample-based

A potential step `V(x) = V0` for `x >= 0` is no polynomial, but it is non-negative, so `PotentialEvolutionSampleBased` applies `e^(-i t V / hbar)` with the sample-based propagator. Run on the Aer simulator, a successful run agrees with the exact phase:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_hamiltonian_simulation.time_independent.sample_based import (
    PotentialEvolutionSampleBased,
)

hbar, t = 1.0, 0.5
x_axis = PositionAxis(size=8, delta_x=0.5, ordering=IndexOrdering.CENTERED)
V = AlgebraicSignal(x_axis, lambda x: np.where(x >= 0, 0.2, 0.0))

evolution = PotentialEvolutionSampleBased(
    V,
    t=t,
    hbar=hbar,
    max_delta=0.02,
    state_preparation_method=SynthesisMethod.DECOMPOSED,
)
assert evolution.num_of_cycles == 20  # t * 4 * 0.2 / hbar = 0.4, in phases of 0.02

psi = Statevector(np.ones(8) / np.sqrt(8))
circuit = QuantumCircuit(*evolution.qregs, *evolution.cregs)
circuit.initialize(psi, evolution.qregs[0])
circuit.compose(evolution, inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
result = simulator.run(
    transpile(circuit, simulator, optimization_level=1), shots=1
).result()
assert result.get_counts().int_outcomes().get(0) == 1  # all cycles succeeded
output = np.asarray(result.get_statevector().data)[:8]

expected = np.exp(-1j * t * V.data / hbar) * psi.data
assert abs(np.vdot(expected, output)) ** 2 > 1 - 1e-6
```

The 20 cycles succeed, and the output of the `psi` register agrees with `e^(-i t V / hbar) psi` to an infidelity of about `1e-7`.

## A wave packet at a potential step

The motion of a wave packet towards the step combines both kinds of evolutions by the first-order splitting `(e^(-i dt T / hbar) e^(-i dt V / hbar))^steps`: the kinetic energy exactly with the direct circuit, the step sample-based. The circuit is built from the registers of `propagator_registers`, and each potential step after the first is guarded by the success flags, so that the flags keep any failure:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)
from qiu_hamiltonian_simulation.time_independent.sample_based import (
    PotentialEvolutionSampleBased,
)
from qiu_quantum_computing.phase_propagator.sample_based import propagator_registers

hbar, mass, num_qubits = 1.0, 1.0, 3
x_axis = PositionAxis(size=2**num_qubits, delta_x=0.75, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)
V = AlgebraicSignal(x_axis, lambda x: np.where(x >= 0, 0.5, 0.0))
T = QuadraticSignal(p_axis, alpha=1 / (2 * mass))
dt, steps = 0.1, 4

potential_step = PotentialEvolutionSampleBased(
    V,
    t=dt,
    hbar=hbar,
    max_delta=0.01,
    state_preparation_method=SynthesisMethod.DECOMPOSED,
)
kinetic_step = MomentumDomainEvolutionQuadratic((-dt / hbar) * T)

x = x_axis.values
packet = np.exp(-((x + 1.5) ** 2) / 2 + 1j * x / hbar)  # moving towards the step
psi = Statevector(packet / np.linalg.norm(packet))

psi_reg, phi_reg, success_flag = propagator_registers(num_qubits)
circuit = QuantumCircuit(psi_reg, phi_reg, success_flag)
circuit.initialize(psi, psi_reg)
for step in range(steps):
    if step == 0:
        circuit.compose(potential_step, inplace=True)
    else:
        with circuit.if_test((success_flag, 0)):
            circuit.compose(potential_step, inplace=True)
    circuit.compose(kinetic_step, psi_reg, inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
result = simulator.run(
    transpile(circuit, simulator, optimization_level=1), shots=1
).result()
assert result.get_counts().int_outcomes().get(0) == 1
output = np.asarray(result.get_statevector().data)[: 2**num_qubits]

# the same splitting with exact phases and NumPy's FFT
expected = psi.data
for _ in range(steps):
    expected = np.exp(-1j * dt * V.data / hbar) * expected
    expected = np.fft.ifft(
        np.exp(-1j * dt * T.data / hbar) * np.fft.fft(expected, norm="ortho"),
        norm="ortho",
    )
assert abs(np.vdot(expected, output)) ** 2 > 1 - 1e-6
```

All 4 potential steps of 20 cycles each succeed, and the output agrees with the same splitting computed with exact phases to an infidelity of about `4e-8`; the remaining error of the splitting itself is of order `dt`.

## A relativistic kinetic energy

The kinetic energy of a relativistic particle, `T(p) = sqrt(p^2 c^2 + m^2 c^4) - m c^2`, is not quadratic in `p`, but it is non-negative, so `KineticEvolutionSampleBased` applies it in the momentum domain, between the Fourier transforms:

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import MomentumAxis, PositionAxis
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_hamiltonian_simulation.time_independent.sample_based import (
    KineticEvolutionSampleBased,
)

hbar, c, mass, t = 1.0, 1.0, 0.5, 0.1
x_axis = PositionAxis(size=8, delta_x=0.5, ordering=IndexOrdering.CENTERED)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)
T = AlgebraicSignal(
    p_axis, lambda p: np.sqrt((p * c) ** 2 + (mass * c**2) ** 2) - mass * c**2
)

evolution = KineticEvolutionSampleBased(
    T,
    t=t,
    hbar=hbar,
    max_delta=0.05,
    state_preparation_method=SynthesisMethod.DECOMPOSED,
    fourier_method=SynthesisMethod.DECOMPOSED,
)

x = x_axis.values
packet = np.exp(-(x**2) / 2 + 1j * x / hbar)
psi = Statevector(packet / np.linalg.norm(packet))
circuit = QuantumCircuit(*evolution.qregs, *evolution.cregs)
circuit.initialize(psi, evolution.qregs[0])
circuit.compose(evolution, inplace=True)
circuit.save_statevector()

simulator = aer_simulator(device="cpu", method="statevector", seed_simulator=1234)
result = simulator.run(
    transpile(circuit, simulator, optimization_level=1), shots=1
).result()
assert result.get_counts().int_outcomes().get(0) == 1
output = np.asarray(result.get_statevector().data)[:8]

expected = np.fft.ifft(
    np.exp(-1j * t * T.data / hbar) * np.fft.fft(psi.data, norm="ortho"),
    norm="ortho",
)
assert evolution.num_of_cycles == 44
assert abs(np.vdot(expected, output)) ** 2 > 1 - 1e-5
```

The kinetic phases, `t sum_k T(p_k) / hbar ~ 2.19` in total, take 44 cycles, and the output agrees with NumPy's FFT evolution to an infidelity of about `2e-6`; since `T(p) = T(-p)`, the direction of the transforms is not tested here, see the [User Guide](user-guide.md#position-and-momentum-domain).
