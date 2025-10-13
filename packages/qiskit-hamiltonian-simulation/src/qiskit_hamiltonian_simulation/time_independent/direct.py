import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.basis_change import QFTGate
from qiskit_phase_propagator.direct import Order2DirectPhase
from qiskit_signals.quantum_signal import QuadraticQuantumSignal

# TODO: add MomentumDomainEvolutionQuadratic


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
