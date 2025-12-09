import numpy as np
import numpy.typing as npt
import qutip as qt
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_encore.initializer import kernel_based_initializer_operator
from qiskit_encore.preparable_statevector import (
    BigUnitaryPreparableStatevector,
    KernelBasedPreparableStatevector,
    PreparableStatevector,
)
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol
from tqdm import tqdm

# def dm_to_ket(rho):
#     eigvals, eigkets = rho.eigenstates()
#     return eigkets[np.argmax(eigvals)]


def apply_phase_protocol(
    psi_in: qt.Qobj,
    signal: ArbitrarySignalForSampleBasedProtocol,
    max_delta: float,
) -> qt.Qobj:
    # calculate the deltas from the signal and the corresponding statevector
    alpha, state = signal.alpha, signal.statevector
    deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)

    # create the initializer for the state
    # U_phi = householder_unitary(state)
    # U_phi_dagger = U_phi.dag()

    # define tools
    N = state.dim
    delta = deltas[0]  # deltas are all the same here
    # ZERO_STATE = qt.basis(N, 0)
    num_cycles = len(deltas)
    phi = qt.Qobj(state.data)

    print(f"number of cycles: {num_cycles}")
    current_state = psi_in.copy()
    for _ in tqdm(range(num_cycles)):
        # TODO: remove initializers and manually tensor product the phi state, the same should happen to the projection step, do not start from the zero state

        current_state = qt.tensor(phi, current_state)

        # Evolve the input state through the circuit
        # TODO: this might be the tricky part, but the operator I think was actually just diagonal so we do not really need to build the full circuit here
        # TODO: let us not even decompose to qubits anymore.
        temp_state = qt.zero_ket([N, N])
        for j in range(N):
            diagonals = np.ones(N, dtype=np.complex128)
            diagonals[j] = np.exp(1j * delta)
            operator = qt.tensor(qt.basis(N, j).proj(), qt.qdiags(diagonals, 0))
            temp_state += operator @ current_state
        current_state = temp_state

        # projection
        projector = qt.basis(N, 0) @ phi.dag()
        operator = qt.tensor(projector, qt.qeye(N))
        current_state = operator @ current_state

        # rho = current_state.ptrace(1)

        # current_state = dm_to_ket(rho)

        # TODO: just directly extract the relevant part of the statevector instead of doing all this projection and tracing out

        # Post-select on the |0...0> outcome of the phi register measurement
        # zero_state_projector = ZERO_STATE.proj()
        # operator = qt.tensor(zero_state_projector, qt.qeye(N))
        # psi_out_post_projection = operator @ psi_out_pre_projection

        # # trace out the phi register
        # print(psi_out_post_projection.dims)
        # test = psi_out_post_projection[0]
        # print(test.dims)
        # psi_out_traced_phi = psi_out_post_projection.proj().ptrace(0)
        # print(psi_out_traced_phi.dims)
        state_array = current_state[0:N]
        current_state = qt.Qobj(state_array)

        # renormalize
        current_state = current_state.unit()

    return current_state


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


def householder_unitary(psi: Statevector) -> qt.Qobj:
    """Construct a unitary U such that U|0> = psi using a Householder reflection."""
    N = psi.dim
    ket0 = qt.basis(N, 0)

    qt_state = qt.Qobj(psi.data)

    # Check if target is already |0>
    if (ket0 - qt_state).norm() < 1e-12:
        return qt.identity(N)

    c = ket0.dag() * qt_state  # <0|psi>
    corrector_phase = np.exp(-1j * np.angle(c))

    psi_phase_corrected = corrector_phase * qt_state

    # Compute the Householder vector
    v = (ket0 - psi_phase_corrected).unit()  # normalized
    U = qt.identity(N) - 2 * v * v.dag()

    U_corrected_global_phase = np.exp(1j * np.angle(c)) * U

    return U_corrected_global_phase
