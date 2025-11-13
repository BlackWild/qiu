"""Module for sample-based phase propagators."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_encore.preparable_statevector import (
    BigUnitaryPreparableStatevector,
    PreparableStatevector,
)
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol


class PhaseProtocolUnitCycleWithoutMeasurement(QuantumCircuit):
    def __init__(
        self,
        delta: float,
        U_phi: QuantumCircuit,
        U_phi_dagger: QuantumCircuit,
    ):
        n = U_phi.num_qubits

        psi_reg = QuantumRegister(n, name=r"\psi")
        phi_reg = QuantumRegister(n, name=r"\phi")

        super().__init__(psi_reg, phi_reg, name="Phase propagator unit")

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


def phase_propagate_one_cycle(
    psi_in: Statevector,
    delta: float,
    phi: PreparableStatevector,
) -> Statevector:
    n = phi.num_qubits

    # Prepare the input state |0...0> ⊗ |psi>
    ZERO_STATE = Statevector.from_label("0" * n)
    # input_state = psi_in.tensor(ZERO_STATE)
    input_state = ZERO_STATE.tensor(psi_in)

    # Create the circuit for one cycle of the phase propagation protocol
    circuit = PhaseProtocolUnitCycleWithoutMeasurement(
        delta=delta,
        U_phi=phi.initializer_circuit,
        U_phi_dagger=phi.de_initializer_circuit,
    )

    # Evolve the input state through the circuit
    psi_out_pre_projection = input_state.evolve(circuit)

    # Post-select on the |0...0> outcome of the phi register measurement
    zero_state_projector = ZERO_STATE.to_operator()

    psi_out_post_projection = psi_out_pre_projection.evolve(
        zero_state_projector, np.arange(n, 2 * n).tolist()
    )

    # renormalize
    normalized_psi_out_post_projection = Statevector(
        psi_out_post_projection.data / np.linalg.norm(psi_out_post_projection.data)
    )

    # obtain the reduced state by tracing out the phi register
    trace_out_rho = partial_trace(
        normalized_psi_out_post_projection, np.arange(n, 2 * n).tolist()
    )  # REMARK: I do not really kno why (0, n) and not (n, 2 * n)
    reduced_state = trace_out_rho.to_statevector()

    return reduced_state


def phase_propagate_state(
    psi_in: Statevector,
    deltas: npt.NDArray | list[float],
    phi: PreparableStatevector,
):
    psi_current = psi_in
    for delta in deltas:
        psi_current = phase_propagate_one_cycle(psi_current, delta, phi)
    return psi_current
