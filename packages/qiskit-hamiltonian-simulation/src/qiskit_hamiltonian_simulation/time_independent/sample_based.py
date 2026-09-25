"""Time evolution under arbitrary potentials and kinetic energies, sample-based.

The evolution `e^(-i t H / hbar)` under a potential `V(x)` or a kinetic energy `T(p)`
of one sign is applied by the sample-based phase propagator of
`qiskit_phase_propagator.sample_based`, on the signal `-t V / hbar` or `-t T / hbar`.
"""

from python_signals.algebraic_signal import SampledSignal
from qiskit.circuit import QuantumCircuit
from qiskit_encore.synthesis_method import SynthesisMethod
from qiskit_phase_propagator.qubit_encoding import num_qubits_of
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
    propagator_registers,
)

from qiskit_hamiltonian_simulation.time_independent.fourier import (
    require_momentum_domain_axis,
    require_position_domain_axis,
    to_momentum_basis,
    to_position_basis,
)


class PotentialEvolutionSampleBased(QuantumCircuit):
    """The evolution `e^(-i t V(x) / hbar)` under a potential of one sign."""

    num_of_cycles: int
    """The number of cycles of the phase propagator."""

    def __init__(
        self,
        V: SampledSignal,
        t: float,
        hbar: float,
        max_delta: float,
        state_preparation_method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the evolution.

        Args:
            V: The potential, a real signal of one sign on a position axis of `2**n`
                samples.
            t: The evolution time.
            hbar: The reduced Planck constant, in the units of `V` and `t`.
            max_delta: The maximum phase per cycle of the phase propagator.
            state_preparation_method: How the preparation of `|phi>` in the phase
                propagator is represented.
        """
        require_position_domain_axis(V.axis)
        psi_reg, phi_reg, success_flag = propagator_registers(num_qubits_of(V.axis))
        super().__init__(psi_reg, phi_reg, success_flag, name="Potential Evolution")

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=(-t / hbar) * V,
            max_delta=max_delta,
            method=state_preparation_method,
        )
        self.num_of_cycles = propagator.num_of_cycles
        self.compose(propagator, inplace=True)


class KineticEvolutionSampleBased(QuantumCircuit):
    """The evolution `e^(-i t T(p) / hbar)` under a kinetic energy of one sign."""

    num_of_cycles: int
    """The number of cycles of the phase propagator."""

    def __init__(
        self,
        T: SampledSignal,
        t: float,
        hbar: float,
        max_delta: float,
        state_preparation_method: SynthesisMethod = SynthesisMethod.GATE,
        fourier_method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the evolution.

        Args:
            T: The kinetic energy, a real signal of one sign on a Fourier domain axis
                of `2**n` samples in the `FFT` ordering.
            t: The evolution time.
            hbar: The reduced Planck constant, in the units of `T` and `t`.
            max_delta: The maximum phase per cycle of the phase propagator.
            state_preparation_method: How the preparation of `|phi>` in the phase
                propagator is represented.
            fourier_method: How the Fourier transforms are represented.
        """
        require_momentum_domain_axis(T.axis)
        num_qubits = num_qubits_of(T.axis)
        psi_reg, phi_reg, success_flag = propagator_registers(num_qubits)
        super().__init__(psi_reg, phi_reg, success_flag, name="Kinetic Evolution")

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=(-t / hbar) * T,
            max_delta=max_delta,
            method=state_preparation_method,
        )
        self.num_of_cycles = propagator.num_of_cycles

        self.compose(
            to_momentum_basis(num_qubits, fourier_method), psi_reg, inplace=True
        )
        self.compose(propagator, inplace=True)
        self.compose(
            to_position_basis(num_qubits, fourier_method), psi_reg, inplace=True
        )
