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


# TODO: maybe it is more performant if I just cut out the 0 projected part instead of using partial trace and stuff!
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

    # TODO: for each phi, this circuit is the same, so you have to cache it before the loop, probably you should actually just convert it to an operator if .evolve() is not caching it and doing the conversion every time

    # TODO: I checked and it seems it does not cache actually and converts it to Operator every time, so you should do it manually outside the loop

    # Create the circuit for one cycle of the phase propagation protocol
    circuit = PhaseProtocolUnitCycleWithoutMeasurement(
        delta=delta,
        U_phi=phi.initializer_circuit,
        U_phi_dagger=phi.de_initializer_circuit,
    )

    print("There!")

    # Evolve the input state through the circuit
    psi_out_pre_projection = input_state.evolve(circuit)

    # TODO: just directly extract the relevant part of the statevector instead of doing all this projection and tracing out

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
) -> Statevector:
    psi_current = psi_in.copy()
    print(f"number of cycles: {len(deltas)}")
    for delta in deltas:
        psi_current = phase_propagate_one_cycle(psi_current, delta, phi)
    return psi_current


def phase_propagate_state_with_constant_delta(
    psi_in: Statevector,
    delta: float,
    num_cycles: int,
    phi: PreparableStatevector,
) -> Statevector:
    initializer = phi.initializer_circuit
    print("There -2!")
    de_initializer = phi.de_initializer_circuit
    print("There -1!")

    circuit = PhaseProtocolUnitCycleWithoutMeasurement(
        delta=delta,
        U_phi=initializer,
        U_phi_dagger=de_initializer,
    )
    print("There 2!")
    operator = Operator(circuit)

    print("There 3!")

    n = phi.num_qubits

    # Prepare the input state |0...0> ⊗ |psi>
    ZERO_STATE = Statevector.from_label("0" * n)

    output_state = psi_in.copy()

    for _ in range(num_cycles):
        # input_state = psi_in.tensor(ZERO_STATE)
        input_state = ZERO_STATE.tensor(psi_in)

        # Evolve the input state through the circuit
        psi_out_pre_projection = input_state.evolve(operator)

        # TODO: just directly extract the relevant part of the statevector instead of doing all this projection and tracing out

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
        output_state = trace_out_rho.to_statevector()

    return output_state


def phase_propagate_state_with_arbitrary_signal(
    psi_in: Statevector,
    signal: ArbitrarySignalForSampleBasedProtocol,
    max_delta: float,
) -> Statevector:
    alpha, state = signal.alpha, signal.statevector
    deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)

    preparable_state = BigUnitaryPreparableStatevector.from_statevector(state)

    print("Here!")

    # psi_out = phase_propagate_state(psi_in, deltas, preparable_state)

    psi_out = phase_propagate_state_with_constant_delta(
        psi_in, deltas[0], len(deltas), preparable_state
    )

    return psi_out


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
