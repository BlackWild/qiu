"""Time evolution under quadratic phases, applied directly by phase circuits."""

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.direct import polynomial_phase_circuit
from qiu_quantum_computing.phase_propagator.qubit_encoding import num_qubits_of
from qiu_signals.algebraic_signal import QuadraticSignal

from qiu_hamiltonian_simulation.time_independent.fourier import (
    require_momentum_domain_axis,
    require_position_domain_axis,
    to_momentum_basis,
    to_position_basis,
)


class PositionDomainEvolutionQuadratic(QuantumCircuit):
    """The unitary `e^(i f(x))` for a quadratic signal `f` on a position axis.

    The basis state `|k>` is multiplied by `e^(i f(x_k))`, e.g. for the potential
    `V` over the time `t`, `f = -t V / hbar`.
    """

    quadratic_signal: QuadraticSignal
    """The quadratic signal `f`."""

    def __init__(self, quadratic_signal: QuadraticSignal) -> None:
        """Initialize the evolution.

        Args:
            quadratic_signal: The quadratic signal on a position axis of `2**n`
                samples.
        """
        require_position_domain_axis(quadratic_signal.axis)
        self.quadratic_signal = quadratic_signal

        psi_reg = QuantumRegister(num_qubits_of(quadratic_signal.axis), name=r"\psi")
        super().__init__(psi_reg, name="Position Domain Evolution")
        self.compose(polynomial_phase_circuit(quadratic_signal), psi_reg, inplace=True)


class MomentumDomainEvolutionQuadratic(QuantumCircuit):
    """The unitary `e^(i f(p))` for a quadratic signal `f` on a momentum axis.

    The state is transformed to the momentum basis, see `fourier`, multiplied by
    `e^(i f(p_k))` and transformed back, e.g. for the kinetic energy `T` over the
    time `t`, `f = -t T / hbar`.
    """

    quadratic_signal: QuadraticSignal
    """The quadratic signal `f`."""

    def __init__(
        self,
        quadratic_signal: QuadraticSignal,
        fourier_method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the evolution.

        Args:
            quadratic_signal: The quadratic signal on a Fourier domain axis of
                `2**n` samples in the `FFT` ordering.
            fourier_method: How the Fourier transforms are represented.
        """
        require_momentum_domain_axis(quadratic_signal.axis)
        self.quadratic_signal = quadratic_signal

        num_qubits = num_qubits_of(quadratic_signal.axis)
        psi_reg = QuantumRegister(num_qubits, name=r"\psi")
        super().__init__(psi_reg, name="Momentum Domain Evolution")

        self.compose(
            to_momentum_basis(num_qubits, fourier_method), psi_reg, inplace=True
        )
        self.compose(polynomial_phase_circuit(quadratic_signal), psi_reg, inplace=True)
        self.compose(
            to_position_basis(num_qubits, fourier_method), psi_reg, inplace=True
        )
