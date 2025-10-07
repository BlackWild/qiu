from qiskit.circuit import QuantumCircuit, Gate, QuantumRegister, ClassicalRegister
import numpy as np
import numpy.typing as npt
from qiskit_encore.quantum_state import PreparableStatevector

class GenericIterativeSampleBasedPhasePropagator(QuantumCircuit):
    """A generic iterative sample-based phase propagator.

    This class implements a generic iterative sample-based phase propagator as a QuantumCircuit.
    It applies a series of quantum operations to simulate the evolution of the phase of a quantum state using a sample-based approach.

    If only one cycle is needed, you can only pass one delta value in the list of deltas.

    Attributes:
        n (int): The number of qubits in the main register.
        number_of_cycles (int): The number of propagation cycles to perform.
        deltas (list[float]): A list of delta values for each cycle.
        U_phi (Gate): The gate representing the unitary operation U_phi.
        U_phi_dagger (Gate): The gate representing the adjoint of U_phi.
    """

    def __init__(
        self,
        deltas: npt.NDArray | list[float],
        U_phi: Gate,
        U_phi_dagger: Gate,
    ) -> None:
        """Initializes the GenericIterativeSampleBasedPhasePropagator with the given parameters.

        Args:
            deltas (deltas: npt.NDArray | list[float]): A list of delta values for each cycle.
            U_phi (Gate): The gate representing the unitary operation U_phi.
            U_phi_dagger (Gate): The gate representing the adjoint of U_phi.
        """
        n = U_phi.num_qubits

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")
        success_flag = ClassicalRegister(n, name="success_flag")


        super().__init__(psi_reg, phi_reg, success_flag, name="Iterative phase propagator")

        number_of_cycles = len(deltas)

        if len(deltas) != number_of_cycles:
            raise ValueError("Length of deltas must be equal to number_of_cycles.")

        # self.n = n
        # self.number_of_cycles = number_of_cycles
        # self.deltas = deltas
        # self.U_phi = U_phi
        # self.U_phi_dagger = U_phi_dagger

        for r in range(number_of_cycles):

            with self.if_test((success_flag, 0)) as else_:
                
                delta = deltas[r]

                # Step 1: initializing the |phi> register
                self.append(U_phi, phi_reg)

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
                self.append(U_phi_dagger, phi_reg)
                self.measure(phi_reg, success_flag)

                # Reset the secondary state to |0> after the Step 3 (partial measurement) of the previous cycle. We expect the result of the measurement to almost always be 0 and if it is not, the protocol has failed. Therefore, this resetting in not strictly required but we are doing it to still see the corrupt output even though an error occurs.
                self.reset(phi_reg)

                # Take a snapshot of the wavefunction at this point
                # self.save_statevector(f"{r}") # type: ignore

            # with else_:
            #     self.metadata["failed_at_cycle"] = r

            # self.metadata = {
            #     "num_of_cycles": number_of_cycles,
            #     "n": n,
            #     "deltas": deltas,
            #     "phi_normalized": phi,
            # }



    @classmethod
    def from_state(cls, state: PreparableStatevector, deltas: npt.NDArray | list[float]) -> "GenericIterativeSampleBasedPhasePropagator":
        """Creates a GenericIterativeSampleBasedPhasePropagator from a given state.

        Args:
            state (PreparableStatevector): The preparable statevector representing the initial state.

        Returns:
            GenericIterativeSampleBasedPhasePropagator: An instance of the propagator initialized with the given state.
        """
        if state.num_qubits is None:
            raise ValueError("The state must have a defined number of qubits.")

        # Create an instance of the propagator
        return GenericIterativeSampleBasedPhasePropagator(
            deltas=deltas,
            U_phi=state.initializer_gate,
            U_phi_dagger=state.de_initializer_gate,
        )
