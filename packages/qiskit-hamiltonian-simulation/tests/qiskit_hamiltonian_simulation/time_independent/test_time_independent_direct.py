"""Unit tests for direct.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    PositionDomainEvolutionQuadratic,
)
from qiskit_pytest_helper.hypothesis_strategies import (
    random_polynomial_signal,
)
from qiskit_signals.quantum_signal import (
    QuadraticQuantumSignal,
)


class TestPositionDomainEvolutionQuadratic:
    """Unit tests for the PositionDomainEvolutionQuadratic circuit."""

    @given(quadratic_signal=random_polynomial_signal(degree=2))
    def test_position_domain_evolution_quadratic(
        self, quadratic_signal: QuadraticQuantumSignal
    ):
        """Tests the PositionDomainEvolutionQuadratic circuit."""
        # Create the quadratic signal
        num_qubits = quadratic_signal.num_qubits

        # Create the circuit
        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("+" * num_qubits)
        circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = PositionDomainEvolutionQuadratic(quadratic_signal)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output = Statevector(circuit)
        expected_output = np.exp(1j * quadratic_signal.data) * psi.data

        assert np.allclose(output.data, expected_output)
