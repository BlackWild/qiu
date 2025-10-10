"""Unit tests for direct.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    PositionDomainEvolutionQuadratic,
)
from qiskit_pytest_helper.hypothesis_strategies import position_axis
from qiskit_signals.quantum_signal import QuadraticQuantumSignal


class TestPositionDomainEvolutionQuadratic:
    """Unit tests for the PositionDomainEvolutionQuadratic circuit."""

    @given(alpha=st.floats(min_value=-1.0, max_value=1.0), x=position_axis())
    def test_position_domain_evolution_quadratic(self, alpha, x):
        """Tests the PositionDomainEvolutionQuadratic circuit."""
        # Create the quadratic signal
        quadratic_signal = QuadraticQuantumSignal(axis=x, alpha=alpha)
        num_qubits = x.num_qubits

        # Create the circuit
        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = PositionDomainEvolutionQuadratic(quadratic_signal)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output = Statevector(circuit)
        expected_output = np.exp(1j * alpha * x.axis_values**2) * psi.data

        assert np.allclose(output.data, expected_output)
