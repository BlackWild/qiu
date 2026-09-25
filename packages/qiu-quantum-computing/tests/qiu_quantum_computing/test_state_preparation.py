"""Unit tests for state_preparation.py."""

import numpy as np
import numpy.typing as npt
import pytest
import qiu_quantum_computing.state_preparation as state_preparation_module
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.assertions import (
    assert_equal_operators,
    assert_equal_states,
    assert_unitary,
)
from qiskit_pytest_helper.circuits import (
    gate_counts,
    transpile_exactly,
    unitary_matrix,
)
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector
from qiu_python_encore.enum import ExtendedEnum
from qiu_qiskit_aer_encore.simulator import aer_simulator
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.state_preparation import (
    decomposed_state_preparation,
    state_preparation_circuit,
    state_preparation_unitary,
)

# A state found by hypothesis, whose preparation by Qiskit's `StatePreparation`
# (qiskit 2.2 to 2.5) has fidelity 0: its isometry synthesis fails when two
# intermediate single-qubit gates are close but not equal. Whether they are depends on
# the rounding of the linear algebra: the synthesis fails with Apple's Accelerate as
# the LAPACK of NumPy, e.g. on macOS, and succeeds with OpenBLAS, e.g. on Linux.
_A = float.fromhex("0x1.6a09e667f3bc7p-2")
_EPS = float.fromhex("0x1.6a09e667f3bc7p-25")
QISKIT_FAILING_STATE = np.array([_A, 1j * _A, _EPS + 1j * _A] + [1j * _A] * 5)
NUMPY_LAPACK = np.show_config(mode="dicts")["Build Dependencies"]["lapack"]["name"]


class FutureSynthesisMethod(ExtendedEnum):
    """The synthesis methods plus one that is not implemented yet."""

    DENSE = "dense"
    GATE = "gate"
    DECOMPOSED = "decomposed"
    FUTURE = "future"


@st.composite
def structured_states(draw, max_qubits: int = 4) -> npt.NDArray[np.complex128]:
    """A strategy for normalized states with exact zeros and (nearly) equal entries.

    Such states hit the degenerate cases of state preparation synthesis.

    Returns:
        The drawn state of 1 to `max_qubits` qubits.
    """
    num_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    base = draw(st.sampled_from([1.0, -1.0, 1j, -1j, (1 + 1j) / np.sqrt(2)]))
    elements = st.one_of(
        st.just(0j),
        st.just(base),
        st.builds(lambda eps: base + eps, st.sampled_from([1e-7, -1e-8, 1e-12])),
        st.complex_numbers(max_magnitude=1, allow_nan=False, allow_infinity=False),
    )
    data = draw(arrays(np.complex128, 2**num_qubits, elements=elements))
    if np.linalg.norm(data) < 1e-3:
        data[0] = 1
    return data / np.linalg.norm(data)


states = st.one_of(
    valid_qiskit_statevector().map(lambda statevector: statevector.data),
    structured_states(),
)
exact_methods = st.sampled_from([SynthesisMethod.DECOMPOSED, SynthesisMethod.DENSE])


def zero_state(num_qubits: int) -> Statevector:
    """The all-zero state."""
    return Statevector.from_int(0, 2**num_qubits)


class TestExactPreparation:
    """Test the methods whose synthesis is done by this package."""

    @given(state=states, method=exact_methods)
    def test_prepares_the_state(self, state, method: SynthesisMethod):
        """Test that the circuit prepares the state exactly, global phase included."""
        circuit = state_preparation_circuit(state, method=method)
        assert_equal_states(circuit, state)

    @given(state=states, method=exact_methods)
    def test_inverse_unprepares_the_state(self, state, method: SynthesisMethod):
        """Test that the inverse circuit maps the state to the all-zero state."""
        circuit = state_preparation_circuit(state, method=method, inverse=True)
        assert_equal_states(
            Statevector(state).evolve(circuit), zero_state(circuit.num_qubits)
        )

    @pytest.mark.parametrize(
        "method", [SynthesisMethod.DECOMPOSED, SynthesisMethod.DENSE]
    )
    def test_state_qiskit_fails_on(self, method: SynthesisMethod):
        """Test the regression state on which Qiskit's synthesis fails."""
        circuit = state_preparation_circuit(QISKIT_FAILING_STATE, method=method)
        assert_equal_states(circuit, QISKIT_FAILING_STATE)


class TestDecomposed:
    """Test the decomposed method."""

    @given(state=states)
    def test_elementary_gates(self, state):
        """Test that only rotations and CNOT gates are used."""
        circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
        assert set(gate_counts(circuit)) <= {"ry", "rz", "cx"}

    @given(state=states)
    def test_cnot_count(self, state):
        """Test the bound of 2**(n+1) - 4 CNOT gates."""
        circuit = decomposed_state_preparation(state)
        assert gate_counts(circuit).get("cx", 0) <= 2 ** (circuit.num_qubits + 1) - 4

    @given(
        state=arrays(
            np.float64, 8, elements=st.floats(min_value=0.1, max_value=1.0)
        ).map(lambda data: data / np.linalg.norm(data))
    )
    def test_real_non_negative_states_need_no_phases(self, state):
        """Test that the phase stage is skipped for real non-negative states."""
        circuit = decomposed_state_preparation(state)
        assert "rz" not in gate_counts(circuit)
        assert circuit.global_phase == 0

    @given(state=states)
    def test_simulates_on_aer(self, state):
        """Test that Aer simulates the circuit to the state."""
        circuit = state_preparation_circuit(state, method=SynthesisMethod.DECOMPOSED)
        circuit.save_statevector()  # type: ignore[attr-defined]
        simulator = aer_simulator(device="cpu", method="statevector")
        result = simulator.run(transpile_exactly(circuit, simulator)).result()
        assert_equal_states(result.get_statevector(), state)


class TestDense:
    """Test the dense method and state_preparation_unitary."""

    @given(state=states)
    def test_unitary(self, state):
        """Test that the matrix is unitary with the state as its first column."""
        unitary = unitary_matrix(state_preparation_unitary(state))
        assert_unitary(unitary)
        assert_equal_states(unitary[:, 0], state)

    def test_single_unitary_gate(self):
        """Test that the circuit is a single unitary gate."""
        circuit = state_preparation_circuit([0.6, 0.8], method=SynthesisMethod.DENSE)
        assert gate_counts(circuit) == {"unitary": 1}


class TestGate:
    """Test the gate method, whose synthesis is left to Qiskit."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("inverse", [False, True])
    def test_prepares_generic_states(self, seed: int, inverse: bool):
        """Test Qiskit's synthesis on generic random states."""
        rng = np.random.default_rng(seed)
        state = rng.normal(size=8) + 1j * rng.normal(size=8)
        state /= np.linalg.norm(state)

        circuit = state_preparation_circuit(
            state, method=SynthesisMethod.GATE, inverse=inverse
        )
        if inverse:
            assert_equal_states(Statevector(state).evolve(circuit), zero_state(3))
        else:
            assert_equal_states(circuit, state)

    def test_is_the_default_method(self):
        """Test that the gate method is the default."""
        circuit = state_preparation_circuit([0.6, 0.8j])
        assert gate_counts(circuit) == {"state_preparation": 1}

    def test_single_high_level_gate(self):
        """Test that the circuit is a single gate, synthesized when transpiling."""
        circuit = state_preparation_circuit([0.6, 0.8], method=SynthesisMethod.GATE)
        assert gate_counts(circuit) == {"state_preparation": 1}

        transpiled = transpile(circuit, basis_gates=["cx", "u"])
        assert set(gate_counts(transpiled)) <= {"cx", "u"}

    @pytest.mark.xfail(
        NUMPY_LAPACK == "accelerate",
        strict=True,
        reason="Qiskit's StatePreparation synthesis is wrong for this state with "
        "Accelerate as the LAPACK. If this passes, Qiskit fixed it; remove this marker.",
    )
    def test_state_qiskit_fails_on(self):
        """Document the Qiskit bug that the decomposed method works around.

        The bug depends on the LAPACK of NumPy: with Accelerate, the test is an expected
        failure, and with others, e.g. OpenBLAS, the preparation is correct.
        """
        circuit = state_preparation_circuit(
            QISKIT_FAILING_STATE, method=SynthesisMethod.GATE
        )
        assert_equal_states(circuit, QISKIT_FAILING_STATE)


class TestValidation:
    """Test that the states are validated."""

    @pytest.mark.parametrize("state", [[1.0, 1.0], [0.0, 0.0], [0.5, 0.5, 0.5, 0.6]])
    def test_rejects_unnormalized_states(self, state):
        """Test that unnormalized states are rejected."""
        with pytest.raises(ValueError, match="normalized"):
            state_preparation_circuit(state)

    @pytest.mark.parametrize("state", [[1.0], [1.0, 0.0, 0.0]])
    def test_rejects_non_qubit_states(self, state):
        """Test that the dimension must be a power of 2 of at least one qubit."""
        with pytest.raises(ValueError, match="qubit"):
            state_preparation_circuit(state)

    def test_accepts_raw_method_values(self):
        """Test that the method can be given as its raw value."""
        circuit = state_preparation_circuit([0.6, 0.8], method="dense")  # type: ignore[arg-type]
        assert gate_counts(circuit) == {"unitary": 1}

    def test_unimplemented_methods_are_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """Test that a newly added method fails loudly until it is implemented."""
        monkeypatch.setattr(
            state_preparation_module, "SynthesisMethod", FutureSynthesisMethod
        )
        with pytest.raises(NotImplementedError, match="future"):
            state_preparation_circuit([0.6, 0.8], method="future")  # type: ignore[arg-type]
