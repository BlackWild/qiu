"""Unit tests for initializer.py."""

import numpy as np
import numpy.typing as npt
from constants import FIDELITY_TOLERANCE, MAX_QUBITS, MIN_QUBITS
from hypothesis import given
from hypothesis import strategies as st
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_encore.initializer import (
    ideal_state_de_initializer,
    ideal_state_initializer,
)
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector


class TestIdealStateInitializer:
    """Tests for ideal_state_initializer and ideal_state_de_initializer."""

    @given(valid_qiskit_statevector(min_qubits=MIN_QUBITS, max_qubits=MAX_QUBITS))
    def test_ideal_state_initializer_no_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_initializer function without transpilation of the state initializer."""
        gate = ideal_state_initializer(statevector, force_transpile=False)
        generated_state = Statevector(gate)
        assert state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector(min_qubits=MIN_QUBITS, max_qubits=MAX_QUBITS))
    def test_ideal_state_initializer_with_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_initializer function with transpilation of the state initializer."""
        gate = ideal_state_initializer(statevector, force_transpile=True)
        generated_state = Statevector(gate)
        assert (
            state_fidelity(generated_state, statevector) >= 1.0 - FIDELITY_TOLERANCE
        ), f"Got {generated_state.data}"

    @given(valid_qiskit_statevector(min_qubits=MIN_QUBITS, max_qubits=MAX_QUBITS))
    def test_ideal_state_de_initializer_no_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_de_initializer function without transpilation of the state de-initializer."""
        circuit = QuantumCircuit(statevector.num_qubits)
        circuit.initialize(statevector.data.tolist(), circuit.qubits)
        gate = ideal_state_de_initializer(statevector, force_transpile=False)
        circuit.append(gate, circuit.qubits)
        de_initialized_state = Statevector(circuit)
        assert abs(de_initialized_state.data[0]) >= 1.0 - FIDELITY_TOLERANCE

    @given(valid_qiskit_statevector(min_qubits=MIN_QUBITS, max_qubits=MAX_QUBITS))
    def test_ideal_state_de_initializer_with_transpile(
        self, statevector: Statevector
    ) -> None:
        """Test the ideal_state_de_initializer function with transpilation of the state de-initializer."""
        circuit = QuantumCircuit(statevector.num_qubits)
        circuit.initialize(statevector.data.tolist(), circuit.qubits)
        gate = ideal_state_de_initializer(statevector, force_transpile=True)
        circuit.append(gate, circuit.qubits)
        de_initialized_state = Statevector(circuit)
        if abs(de_initialized_state.data[0]) < 1.0 - FIDELITY_TOLERANCE:
            circuit.draw()
        assert abs(de_initialized_state.data[0]) >= 1.0 - FIDELITY_TOLERANCE

    @given(
        statevector=valid_qiskit_statevector(
            min_qubits=MIN_QUBITS, max_qubits=MAX_QUBITS
        ),
        force_transpilation=st.sampled_from([True]),
    )
    def test_initializer_aer_simulatable(
        self, statevector: Statevector, force_transpilation: bool
    ) -> None:
        """Test that the initializer gate can be transpiled for AerSimulator."""
        state = statevector
        circuit = QuantumCircuit(state.num_qubits)
        circuit.append(
            ideal_state_initializer(state),
            circuit.qubits,
        )

        simulator = AerSimulator()
        transpiled = transpile(circuit, simulator) if force_transpilation else circuit

        job = simulator.run(transpiled, shots=1)
        result = job.result()

        assert result.success

    @given(
        statevector=valid_qiskit_statevector(min_qubits=1, max_qubits=5),
        force_transpilation=st.sampled_from([True]),
    )
    def test_de_initializer_aer_simulatable(
        self, statevector: Statevector, force_transpilation: bool
    ) -> None:
        """Test that the de-initializer gate can be transpiled for AerSimulator."""
        circuit = QuantumCircuit(statevector.num_qubits)
        circuit.initialize(statevector.data.tolist(), circuit.qubits)
        circuit.append(
            ideal_state_de_initializer(statevector, True),
            circuit.qubits,
        )

        simulator = AerSimulator()
        transpiled = transpile(circuit, simulator) if force_transpilation else circuit

        job = simulator.run(transpiled, shots=1)
        result = job.result()

        assert result.success
