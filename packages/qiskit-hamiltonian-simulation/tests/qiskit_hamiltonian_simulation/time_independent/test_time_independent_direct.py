"""Unit tests for direct.py."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
    PositionDomainEvolutionQuadratic,
)
from qiskit_pytest_helper.hypothesis_strategies import (
    random_polynomial_signal,
)
from qiskit_signals.helper_types import AxisType, EncodingType
from qiskit_signals.quantum_signal import (
    QuadraticQuantumSignal,
)


class TestPositionDomainEvolutionQuadratic:
    """Unit tests for the PositionDomainEvolutionQuadratic circuit."""

    @given(
        quadratic_signal=random_polynomial_signal(degree=2, axis_type=AxisType.POSITION)
    )
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


class TestMomentumDomainEvolutionQuadratic:
    """Unit tests for the MomentumDomainEvolutionQuadratic circuit."""

    @given(
        quadratic_signal=random_polynomial_signal(
            degree=2,
            axis_type=AxisType.MOMENTUM,
            forced_encoding=EncodingType.TWOS_COMPLEMENT,
        )
    )
    def test_momentum_domain_evolution_quadratic(
        self, quadratic_signal: QuadraticQuantumSignal
    ):
        """Tests the MomentumDomainEvolutionQuadratic circuit."""
        # Create the quadratic signal
        num_qubits = quadratic_signal.num_qubits

        # Create the circuit
        circuit = QuantumCircuit(num_qubits)
        psi = Statevector.from_label("0" * num_qubits)
        # REMARK: no initialization to have a flat fourier spectrum
        # circuit.initialize(psi.data.tolist(), circuit.qubits)
        propagator = MomentumDomainEvolutionQuadratic(quadratic_signal)
        circuit.compose(propagator, circuit.qubits, inplace=True)

        output = Statevector(circuit)

        # Fourier transform of psi.data and output.data
        psi_fourier = np.fft.fft(psi.data)
        output_fourier = np.fft.fft(output.data)

        expected_output = np.exp(1j * quadratic_signal.data) * psi_fourier

        assert np.allclose(output_fourier, expected_output)
