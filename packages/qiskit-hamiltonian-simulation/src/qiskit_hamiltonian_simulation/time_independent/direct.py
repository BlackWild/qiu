"""Module implementing time-independent evolution under several polynomial Hamiltonians using direct phase application."""

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.basis_change import QFTGate
from qiskit_phase_propagator.direct import Order2DirectPhase
from qiskit_signals.quantum_axis import MomentumAxis, PositionAxis
from qiskit_signals.quantum_signal import QuadraticQuantumSignal

# TODO: move these classes to qiskit-phase-propagator, only keep things related to actual terms in the Hamiltonian here; like potential and kinetic energy terms


class PositionDomainEvolutionQuadratic(QuantumCircuit):
    """A quantum circuit evolving a state under a quadratic phase profile.

    This circuit implements the unitary operation exp(-i * alpha * X^2) where X is the
    position operator.

    Attributes:
        QuadraticQuantumSignal (QuadraticQuantumSignal): The quadratic quantum signal representing the phase profile.
    """

    quadratic_signal: QuadraticQuantumSignal

    def __init__(
        self,
        quadratic_signal: QuadraticQuantumSignal,
    ) -> None:
        """Initializes the PositionDomainEvolution with the given parameters."""
        if not quadratic_signal.axis.is_fourier_domain_axis:
            raise ValueError(
                "The axis of the quadratic signal must be a Fourier domain axis."
            )

        self.quadratic_signal = quadratic_signal

        n = quadratic_signal.axis.num_qubits

        psi_reg = QuantumRegister(n, name=r"\psi")
        super().__init__(psi_reg, name="Position Domain Evolution")

        propagator = Order2DirectPhase(
            coef=quadratic_signal.effective_alpha,
            num_qubits=n,
            encoding=quadratic_signal.encoding,
        )

        self.compose(propagator, psi_reg, inplace=True)


class MomentumDomainEvolutionQuadratic(QuantumCircuit):
    """A quantum circuit evolving a state under a quadratic phase profile in momentum space.

    This circuit implements the unitary operation exp(-i * alpha * P^2) where P is the
    momentum operator.

    Attributes:
        QuadraticQuantumSignal (QuadraticQuantumSignal): The quadratic quantum signal representing the phase profile.
    """

    quadratic_signal: QuadraticQuantumSignal

    def __init__(
        self,
        quadratic_signal: QuadraticQuantumSignal,
    ) -> None:
        """Initializes the MomentumDomainEvolution with the given parameters."""
        if not quadratic_signal.axis.is_fourier_domain_axis:
            raise ValueError(
                "The axis of the quadratic signal must be a Fourier domain axis."
            )

        self.quadratic_signal = quadratic_signal

        n = quadratic_signal.axis.num_qubits

        psi_reg = QuantumRegister(n, name=r"\psi")
        super().__init__(psi_reg, name="Momentum Domain Evolution")

        propagator = Order2DirectPhase(
            coef=quadratic_signal.effective_alpha,
            num_qubits=n,
            encoding=quadratic_signal.encoding,
        )

        qft = QFTGate(n)
        iqft = qft.inverse()

        self.append(qft, psi_reg)
        self.compose(propagator, psi_reg, inplace=True)
        self.append(iqft, psi_reg)
