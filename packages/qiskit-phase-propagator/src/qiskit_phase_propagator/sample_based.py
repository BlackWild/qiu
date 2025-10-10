"""Module for sample-based phase propagators."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_encore.preparable_statevector import (
    IdeallyPreparableStatevector,
    PreparableStatevector,
)
from qiskit_signals.quantum_signal import (
    QuadraticQuantumSignal,
)


class GenericIterativeSampleBasedPhasePropagator(QuantumCircuit):
    """A generic iterative sample-based phase propagator.

    This class implements a generic iterative sample-based phase propagator as a QuantumCircuit.
    It applies a series of quantum operations to simulate the evolution of the phase of a quantum state using a sample-based approach.

    If only one cycle is needed, you can only pass one delta value in the list of deltas.
    """

    def __init__(
        self,
        deltas: npt.NDArray | list[float],
        U_phi: QuantumCircuit,
        U_phi_dagger: QuantumCircuit,
        take_snapshot: bool = False,
    ) -> None:
        """Initializes the GenericIterativeSampleBasedPhasePropagator with the given parameters.

        Args:
            deltas (deltas: npt.NDArray | list[float]): A list of delta values for each cycle.
            U_phi (QuantumCircuit): The circuit representing the unitary operation U_phi.
            U_phi_dagger (QuantumCircuit): The circuit representing the adjoint of U_phi.
            take_snapshot (bool, optional): Whether to take snapshots of the wavefunction at each cycle. Default is False. If True, the statevector is saved at each cycle with the label being the cycle index.
        """
        n = U_phi.num_qubits

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")

        super().__init__(
            psi_reg, phi_reg, success_flag, name="Iterative phase propagator"
        )

        number_of_cycles = len(deltas)

        for r in range(number_of_cycles):
            with self.if_test((success_flag, 0)) as else_:  # noqa: F841, TODO: remove if not used
                delta = deltas[r]

                # Step 1: initializing the |phi> register
                self.compose(U_phi, phi_reg, inplace=True)

                # Step 2: the partial phase operator
                # Flag qubit computation
                for i in range(psi_reg.size):
                    self.cx(
                        phi_reg[psi_reg.size - 1 - i],
                        psi_reg[psi_reg.size - 1 - i],
                        ctrl_state=0,
                    )

                # The application of the phase
                self.mcp(delta, psi_reg[0:-1], psi_reg[-1])

                # "Un-computing" the flag qubits
                for i in range(psi_reg.size):
                    self.cx(phi_reg[i], psi_reg[i], ctrl_state=0)

                # Step 3: partial measurement of the secondary register
                self.compose(U_phi_dagger, phi_reg, inplace=True)
                self.measure(phi_reg, success_flag)

                # Reset the secondary state to |0> after the Step 3 (partial measurement) of the previous cycle. We expect the result of the measurement to almost always be 0 and if it is not, the protocol has failed. Therefore, this resetting in not strictly required but we are doing it to still see the corrupt output even though an error occurs.
                self.reset(phi_reg)

                # Take a snapshot of the wavefunction at this point
                if take_snapshot:
                    self.save_statevector(f"{r}")  # type: ignore

            # with else_:
            #     self.metadata["failed_at_cycle"] = r

            # self.metadata = {
            #     "num_of_cycles": number_of_cycles,
            #     "n": n,
            #     "deltas": deltas,
            #     "phi_normalized": phi,
            # }

    @classmethod
    def from_state(
        cls, state: PreparableStatevector, deltas: npt.NDArray | list[float]
    ) -> "GenericIterativeSampleBasedPhasePropagator":
        """Creates a GenericIterativeSampleBasedPhasePropagator from a given state.

        Args:
            state (PreparableStatevector): The preparable statevector representing the initial state.
            deltas (npt.NDArray | list[float]): A list of delta values for each cycle.

        Returns:
            GenericIterativeSampleBasedPhasePropagator: An instance of the propagator initialized with the given state.
        """
        if state.num_qubits is None:
            raise ValueError("The state must have a defined number of qubits.")

        # Create an instance of the propagator
        return GenericIterativeSampleBasedPhasePropagator(
            deltas=deltas,
            U_phi=state.initializer_circuit,
            U_phi_dagger=state.de_initializer_circuit,
        )


class QuadraticSignalSampleBasedPhasePropagator(QuantumCircuit):
    """A quadratic signal phase propagator.

    This class implements a quadratic signal phase propagator as a QuantumCircuit.
    It applies a series of quantum operations to simulate the evolution of the phase of a quantum state using a quadratic signal approach.

    Attributes:
        delta (float): The delta value for the propagation.
        U_phi (QuantumCircuit): The circuit representing the unitary operation U_phi.
        U_phi_dagger (QuantumCircuit): The circuit representing the adjoint of U_phi.
    """

    def __init__(
        self,
        signal: QuadraticQuantumSignal,
        max_delta: float,
    ) -> None:
        """Initializes the QuadraticSignalPhasePropagator with the given parameters.

        Args:
            signal (QuadraticQuantumSignal): The quadratic quantum signal containing alpha and statevector.
            max_delta (float): The maximum delta value for slicing the alpha value.
        """
        n = signal.num_qubits
        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")
        super().__init__(
            psi_reg, phi_reg, success_flag, name="Quadratic signal phase propagator"
        )

        alpha, state = signal.alpha, signal.statevector
        deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)

        preparable_state = IdeallyPreparableStatevector(state.data, normalize=True)

        propagator = GenericIterativeSampleBasedPhasePropagator.from_state(
            state=preparable_state, deltas=deltas
        )

        self.compose(propagator, inplace=True)


def slice_alpha_to_deltas(alpha: float, delta: float) -> npt.NDArray:
    """Slices the alpha value into a list of deltas, each with a value of delta except possibly the last one.

    Args:
        alpha (float): The total alpha value to be sliced.
        delta (float): The maximum value for each delta slice.

    Returns:
        npt.NDArray: An array of delta values that sum up to alpha.
    """

    # assure delta is positive
    if delta <= 0:
        raise ValueError("The delta must be positive.")

    delta *= np.sign(alpha)

    full_deltas = alpha // delta
    remaining_alpha = alpha - int(full_deltas) * delta
    deltas = np.concatenate((delta * np.ones(int(full_deltas)), [remaining_alpha]))

    return deltas


def slice_alpha_to_deltas_evenly(alpha: float, max_delta: float) -> npt.NDArray:
    """Slices the alpha value into a list of deltas, each with a maximum value of max_delta.

    Args:
        alpha (float): The total alpha value to be sliced.
        max_delta (float): The maximum value for each delta slice.

    Returns:
        npt.NDArray: An array of delta values that sum up to alpha.
    """

    # assure delta is positive
    if max_delta <= 0:
        raise ValueError("The max_delta must be positive.")

    max_delta *= np.sign(alpha)

    number_of_deltas = int(np.ceil(np.abs(alpha / max_delta)))
    delta = alpha / number_of_deltas
    deltas = delta * np.ones(number_of_deltas)

    return deltas
