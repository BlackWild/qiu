"""Module for time-independent Hamiltonian simulation."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library.basis_change import QFTGate
from qiskit_phase_propagator.sample_based import (
    QuadraticQuantumSignal,
    QuadraticSignalSampleBasedPhasePropagator,
)

from qiskit_hamiltonian_simulation.helper_types import RealSignalType


class PositionDomainEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a position-domain Hamiltonian."""

    def __init__(
        self,
        f_of_x: RealSignalType,
        x: npt.NDArray | list[float],
        max_delta: float,
    ) -> None:
        """Initializes the PositionDomainEvolution with the given circuit."""

        n = int(np.ceil(np.log2(len(x))))

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(
            psi_reg, phi_reg, success_flag, name="Position Domain Evolution"
        )

        x_array = np.asarray(x, dtype=np.float64)
        signal_data = f_of_x(x_array)
        signal = QuadraticQuantumSignal.from_data(signal_data)

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )

        self.compose(propagator, inplace=True)


class MomentumDomainEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a momentum-domain Hamiltonian."""

    def __init__(
        self,
        f_of_p: RealSignalType,
        p: npt.NDArray | list[float],
        max_delta: float,
    ) -> None:
        """Initializes the MomentumDomainEvolution with the given circuit."""
        n = int(np.ceil(np.log2(len(p))))

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(
            psi_reg, phi_reg, success_flag, name="Momentum Domain Evolution"
        )

        p_array = np.asarray(p, dtype=np.float64)
        signal_data = f_of_p(p_array)
        signal = QuadraticQuantumSignal.from_data(signal_data)

        qft = QFTGate(signal.num_qubits)
        iqft = qft.inverse()

        self.append(qft)

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )

        self.append(iqft)

        self.compose(propagator, inplace=True)


class PotentialEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a potential energy operator."""

    def __init__(
        self,
        V: RealSignalType,
        x: npt.NDArray | list[float],
        t: float,
        hbar: float,
        max_delta: float,
    ) -> None:
        """Initializes the PotentialEvolution with the given circuit."""
        n = int(np.ceil(np.log2(len(x))))
        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(psi_reg, phi_reg, success_flag, name="Potential Evolution")

        x_array = np.asarray(x, dtype=np.float64)
        signal_data = -1 / hbar * V(x_array) * t
        signal = QuadraticQuantumSignal.from_data(signal_data)

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )

        self.compose(propagator, inplace=True)


class KineticEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a kinetic energy operator."""

    def __init__(
        self,
        T: RealSignalType,
        p: npt.NDArray | list[float],
        t: float,
        hbar: float,
        max_delta: float,
    ) -> None:
        """Initializes the KineticEvolution with the given circuit."""
        n = int(np.ceil(np.log2(len(p))))
        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(psi_reg, phi_reg, success_flag, name="Kinetic Evolution")

        p_array = np.asarray(p, dtype=np.float64)
        signal_data = -1 / hbar * T(p_array) * t
        signal = QuadraticQuantumSignal.from_data(signal_data)

        qft = QFTGate(signal.num_qubits)
        iqft = qft.inverse()

        self.append(qft)

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )

        self.append(iqft)

        self.compose(propagator, inplace=True)
