"""Unit tests for qft.py."""

import numpy as np
import pytest
import qiu_quantum_computing.qft as qft_module
from hypothesis import given
from hypothesis import strategies as st
from qiskit import transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_pytest_helper.assertions import (
    assert_equal_operators,
    assert_equal_states,
    assert_unitary,
)
from qiskit_pytest_helper.circuits import gate_counts
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector
from qiu_python_encore.enum import ExtendedEnum
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.qft import qft_circuit, qft_matrix


class FutureSynthesisMethod(ExtendedEnum):
    """The synthesis methods plus one that is not implemented yet."""

    DENSE = "dense"
    GATE = "gate"
    DECOMPOSED = "decomposed"
    FUTURE = "future"


all_methods = pytest.mark.parametrize("method", list(SynthesisMethod))
num_qubits = pytest.mark.parametrize("n", range(1, 6))


class TestQFTMatrix:
    """Test qft_matrix."""

    @num_qubits
    def test_unitary(self, n: int):
        """Test that the matrix is unitary."""
        assert_unitary(qft_matrix(n))

    @num_qubits
    def test_is_qiskits_qft(self, n: int):
        """Test that the matrix is the one of Qiskit's QFTGate."""
        assert_equal_operators(qft_matrix(n), QFTGate(n))


class TestQFTCircuit:
    """Test qft_circuit."""

    @all_methods
    @num_qubits
    def test_implements_the_qft(self, method: SynthesisMethod, n: int):
        """Test that every method implements the QFT and its inverse."""
        expected = Operator(qft_matrix(n))
        assert_equal_operators(qft_circuit(n, method=method), expected)
        assert_equal_operators(
            qft_circuit(n, inverse=True, method=method), expected.adjoint()
        )

    @given(
        statevector=valid_qiskit_statevector(), method=st.sampled_from(SynthesisMethod)
    )
    def test_numpy_fft_convention(
        self, statevector: Statevector, method: SynthesisMethod
    ):
        """Test that the QFT is NumPy's orthonormal inverse DFT, and vice versa."""
        assert statevector.num_qubits is not None
        n, data = statevector.num_qubits, statevector.data

        assert_equal_states(
            statevector.evolve(qft_circuit(n, method=method)),
            np.fft.ifft(data, norm="ortho"),
        )
        assert_equal_states(
            statevector.evolve(qft_circuit(n, inverse=True, method=method)),
            np.fft.fft(data, norm="ortho"),
        )

    def test_decomposed_gates(self):
        """Test that the decomposed QFT uses elementary gates."""
        circuit = qft_circuit(4, method=SynthesisMethod.DECOMPOSED)
        assert set(gate_counts(circuit)) <= {"h", "cp", "swap"}

    def test_gate_is_the_default_method(self):
        """Test that the gate method is the default."""
        assert gate_counts(qft_circuit(3)) == {"qft": 1}

    def test_gate_is_left_to_qiskit(self):
        """Test that the gate method is a single QFTGate, synthesized on transpiling."""
        circuit = qft_circuit(4, method=SynthesisMethod.GATE)
        assert gate_counts(circuit) == {"qft": 1}
        assert set(gate_counts(transpile(circuit, basis_gates=["cx", "u"]))) <= {
            "cx",
            "u",
        }

    def test_dense_is_one_unitary(self):
        """Test that the dense QFT is a single unitary gate."""
        assert gate_counts(qft_circuit(3, method=SynthesisMethod.DENSE)) == {
            "unitary": 1
        }

    def test_invalid_number_of_qubits(self):
        """Test that at least one qubit is required."""
        with pytest.raises(ValueError, match="at least 1"):
            qft_circuit(0)

    def test_unimplemented_methods_are_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """Test that a newly added method fails loudly until it is implemented."""
        monkeypatch.setattr(qft_module, "SynthesisMethod", FutureSynthesisMethod)
        with pytest.raises(NotImplementedError, match="future"):
            qft_circuit(2, method="future")  # type: ignore[arg-type]
