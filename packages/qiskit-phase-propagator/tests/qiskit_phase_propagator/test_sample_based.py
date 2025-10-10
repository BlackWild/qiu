"""Unit tests for sample_based.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit_aer_encore.simulator import generate_aer_simulator
from qiskit_encore.preparable_statevector import IdeallyPreparableStatevector
from qiskit_phase_propagator.sample_based import (
    GenericIterativeSampleBasedPhasePropagator,
)
from qiskit_pytest_helper.hypothesis_strategies import state_pairs_with_equal_qubits


class TestGenericIterativeSampleBasedPhasePropagator:
    """Test the GenericIterativeSampleBasedPhasePropagator."""

    @given(
        states=state_pairs_with_equal_qubits(),
        deltas=st.lists(
            st.floats(min_value=0.0, max_value=0.1), min_size=1, max_size=4
        ),
    )
    def test_essentials(
        self, states: tuple[Statevector, Statevector], deltas: list[float]
    ):
        """Test the essentials of the GenericIterativeSampleBasedPhasePropagator."""
        psi, phi = states
        preparable_state = IdeallyPreparableStatevector.from_statevector(phi)
        assert psi.num_qubits == phi.num_qubits == preparable_state.num_qubits
        n = preparable_state.num_qubits

        propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
            preparable_state, deltas
        )
        assert propagator.num_qubits == 2 * n

    @given(
        states=state_pairs_with_equal_qubits(),
        deltas=st.lists(
            st.floats(min_value=0.0, max_value=0.1), min_size=1, max_size=4
        ),
    )
    def test_correct_phase_application(
        self, states: tuple[Statevector, Statevector], deltas: list[float]
    ):
        """Test that the GenericIterativeSampleBasedPhasePropagator applies the correct phase."""
        psi, phi = states
        preparable_state = IdeallyPreparableStatevector.from_statevector(phi)
        assert psi.num_qubits == phi.num_qubits == preparable_state.num_qubits
        n = preparable_state.num_qubits

        propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
            preparable_state, deltas
        )

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        circuit = QuantumCircuit(psi_reg, phi_reg, success_flag)
        circuit.initialize(psi.data.tolist(), psi_reg)
        circuit.compose(propagator, circuit.qubits, circuit.clbits, inplace=True)
        circuit.save_statevector()  # type: ignore

        simulator = generate_aer_simulator()
        transpiled = transpile(circuit, simulator)
        job = simulator.run(transpiled, shots=1)
        result = job.result()
        counts: dict[int, int] = result.get_counts(circuit).int_outcomes()
        output_state_full = result.get_statevector(circuit)

        # Check that all measured qubits are 0
        assert counts[0] == 1

        # Check the output state
        traced = partial_trace(
            output_state_full,
            np.arange(
                preparable_state.num_qubits, 2 * preparable_state.num_qubits
            ).tolist(),
        )
        output_state = traced.to_statevector()
        alpha = np.sum(deltas)
        expected_output = np.exp(1j * alpha * np.abs(phi.data) ** 2) * psi.data
        assert np.isclose(
            state_fidelity(output_state.data.tolist(), expected_output), 1.0
        )
