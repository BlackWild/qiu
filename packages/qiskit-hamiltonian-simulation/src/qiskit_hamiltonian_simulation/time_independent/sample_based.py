"""Module for time-independent Hamiltonian simulation."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library.basis_change import QFTGate
from qiskit_phase_propagator.sample_based import (
    QuadraticSignalSampleBasedPhasePropagator,
)
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol


class PotentialEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a potential energy operator."""

    num_of_cycles: int

    def __init__(
        self,
        V: ArbitrarySignalForSampleBasedProtocol,
        t: float,
        hbar: float,
        max_delta: float,
    ) -> None:
        """Initializes the PotentialEvolution with the given circuit."""
        n = V.num_qubits
        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(psi_reg, phi_reg, success_flag, name="Potential Evolution")

        signal = (-1 / hbar * t) * V

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )

        self.num_of_cycles = propagator.num_of_cycles

        self.compose(propagator, inplace=True)


class KineticEvolutionSampleBased(QuantumCircuit):
    """A quantum circuit representing the evolution under a kinetic energy operator."""

    num_of_cycles: int

    def __init__(
        self,
        T: ArbitrarySignalForSampleBasedProtocol,
        t: float,
        hbar: float,
        max_delta: float,
    ):
        """Initializes the KineticEvolution with the given circuit."""
        n = T.num_qubits
        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(psi_reg, phi_reg, success_flag, name="Kinetic Evolution")

        signal = (-1 / hbar * t) * T

        propagator = QuadraticSignalSampleBasedPhasePropagator(
            signal=signal,
            max_delta=max_delta,
        )
        self.num_of_cycles = propagator.num_of_cycles

        qft = QFTGate(n)
        iqft = qft.inverse()

        self.append(qft, psi_reg)
        self.compose(propagator, inplace=True)
        self.append(iqft, psi_reg)
