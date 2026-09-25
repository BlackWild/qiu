# Examples

Each example is a complete test, called directly at the end of its block as pytest would call it.

## A property-based test of a state preparation

The robust state preparation of [qiu-quantum-computing](../qiu-quantum-computing/index.md) must prepare every state exactly, including its global phase, whichever way it is synthesized. `valid_qiskit_statevector` generates normalized states of 2 to 4 qubits, here at most 3 to keep the decomposed circuits small, and `assert_equal_states` compares the state the circuit prepares from `|0...0>` with the generated one.

```python
from hypothesis import given
from hypothesis import strategies as st
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


@given(
    state=valid_qiskit_statevector(max_qubits=3),
    method=st.sampled_from(list(SynthesisMethod)),
)
def test_prepares_the_state(state: Statevector, method: SynthesisMethod):
    """Test that the preparation circuit prepares the state from |0...0>."""
    circuit = PreparableState(state, method).circuit
    assert circuit.num_qubits == state.num_qubits
    assert_equal_states(circuit, state)


test_prepares_the_state()
```

Hypothesis runs the test on 100 generated states and synthesis methods, and every preparation is exact within Qiskit's default tolerances.

## Operators with and without their global phase

`assert_equal_operators` compares a circuit with a matrix, global phase included. Qiskit's `QFTGate` is the orthonormal inverse DFT of the amplitudes, `numpy.fft.ifft(..., norm="ortho")`, and is unitary. `RZ(theta)` and `P(theta)`, however, differ by the global phase `e^(-i theta / 2)`: they are equivalent, but not equal, and would differ as controlled gates.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
from qiskit_pytest_helper.assertions import assert_equal_operators, assert_unitary
from qiskit_pytest_helper.circuits import unitary_matrix


def test_qft_is_the_inverse_dft():
    """Test the QFT against the matrix of numpy's orthonormal inverse DFT."""
    dimension = 2**3
    dft = np.fft.ifft(np.eye(dimension), axis=0, norm="ortho")
    assert_equal_operators(QFTGate(3), dft)
    assert_unitary(dft)


def test_rz_is_a_phase_gate_up_to_a_global_phase():
    """Test that RZ and P only agree up to their global phase."""
    rz, phase = QuantumCircuit(1), QuantumCircuit(1)
    rz.rz(0.4, 0)
    phase.p(0.4, 0)

    assert Operator(rz).equiv(Operator(phase))
    try:
        assert_equal_operators(rz, phase)
    except AssertionError as error:
        assert "equal" in str(error)  # "up to a global phase, they are equal"
    else:
        raise AssertionError("RZ and P must not be equal.")
    assert_equal_operators(np.exp(-0.2j) * unitary_matrix(phase), rz)


test_qft_is_the_inverse_dft()
test_rz_is_a_phase_gate_up_to_a_global_phase()
```

The QFT matches the matrix exactly, while the comparison of `RZ` and `P` fails with the message that they are equal up to a global phase; with the phase `e^(-0.2 i)` multiplied in, they are equal.

## Transpiling without dropping small rotations

Circuits simulated on Aer must be transpiled for the simulator. At optimization level 2 and above, the transpiler removes rotations it deems equivalent to the identity; `transpile_exactly` transpiles at level 1, which keeps them.

```python
from qiskit import QuantumCircuit, transpile
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiskit_pytest_helper.assertions import assert_equal_operators
from qiskit_pytest_helper.circuits import (
    EXACT_OPTIMIZATION_LEVEL,
    gate_counts,
    transpile_exactly,
)


def test_small_rotations_survive_the_transpilation():
    """Test that a rotation by 1e-6 survives, unlike at optimization level 2."""
    circuit = QuantumCircuit(1)
    circuit.rx(1e-6, 0)
    simulator = aer_simulator(device="cpu", method="statevector")

    exact = transpile_exactly(circuit, simulator)
    assert EXACT_OPTIMIZATION_LEVEL == 1
    assert gate_counts(exact) == {"rx": 1}
    assert_equal_operators(exact, circuit)

    optimized = transpile(circuit, simulator, optimization_level=2)
    assert gate_counts(optimized) == {}
    try:
        assert_equal_operators(optimized, circuit)
    except AssertionError:
        pass  # the dropped rotation changes the unitary by 5e-7
    else:
        raise AssertionError("The optimized circuit must differ.")


test_small_rotations_survive_the_transpilation()
```

The exactly transpiled circuit keeps its `rx` gate and its unitary, while level 2 returns an empty circuit whose unitary differs by `sin(5e-7)` in the off-diagonal entries, far above Qiskit's absolute tolerance of `1e-8`.

## A sample-based propagator against its closed form

A sample-based propagator of [qiu-quantum-computing](../qiu-quantum-computing/index.md) is simulated on Aer by `run_propagator`, which returns whether all cycles succeeded and the final amplitudes of the `psi` register. Successful runs must match `exact_cycles` exactly, global phase included; failed ones are discarded with `assume`. `state_pairs_with_equal_qubits` generates `psi` and `phi` of the same number of qubits.

```python
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    GenericIterativeSampleBasedPhasePropagator,
)
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import state_pairs_with_equal_qubits
from qiskit_pytest_helper.propagation import exact_cycles, run_propagator


@settings(max_examples=10, deadline=None)
@given(
    states=state_pairs_with_equal_qubits(max_qubits=3),
    deltas=st.lists(st.floats(min_value=0.0, max_value=0.1), min_size=1, max_size=3),
)
def test_applies_the_cycles(states: tuple[Statevector, Statevector], deltas):
    """Test that each successful cycle applies its exact map to psi."""
    psi, phi = states
    propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
        PreparableState(phi, method=SynthesisMethod.DENSE), deltas
    )
    assert propagator.num_qubits == 2 * psi.num_qubits

    succeeded, output = run_propagator(propagator, psi)
    assume(succeeded)
    assert_equal_states(output, exact_cycles(psi.data, phi.data, deltas))


test_applies_the_cycles()
```

With small `delta`s the cycles almost always succeed, and the simulated amplitudes equal the closed form within Qiskit's tolerances. The simulations are expensive, hence the fewer examples and no deadline.

## A phase signal on a qubit axis

Signals on qubit axes combine `qubit_axes` with the signal strategies of [python-pytest-helper](../python-pytest-helper/index.md). The direct phase circuit of a polynomial signal must multiply each basis state `|k>` by `e^(i f(x_k))`, so the state it prepares from the uniform superposition is known exactly. `moderate_alphas` keeps the phases of the monomials moderate.

```python
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.hypothesis_strategies import monomial_signals
from qiu_signals.algebraic_signal import PolynomialSignal
from qiu_signals.physical_axis import AxisDomain
from qiskit.quantum_info import Statevector
from qiu_quantum_computing.phase_propagator.direct import polynomial_phase_circuit
from qiskit_pytest_helper.assertions import assert_equal_states
from qiskit_pytest_helper.hypothesis_strategies import moderate_alphas, qubit_axes


@given(
    signal=monomial_signals(
        qubit_axes(AxisDomain.POSITION),
        alphas=moderate_alphas,
        powers=st.integers(min_value=1, max_value=3),
    )
)
def test_applies_the_signal(signal: PolynomialSignal):
    """Test that the phase circuit applies e^(i f) to the uniform superposition."""
    circuit = polynomial_phase_circuit(signal)
    num_qubits = circuit.num_qubits
    assert 2**num_qubits == signal.axis.size

    uniform = Statevector.from_label("+" * num_qubits)
    expected = np.exp(1j * np.asarray(signal.data)) * uniform.data
    assert_equal_states(uniform.evolve(circuit), expected)


test_applies_the_signal()
```

For monomials of powers 1 to 3 on position axes of 4 to 16 samples in all orderings, the circuit applies the sampled phases exactly.
