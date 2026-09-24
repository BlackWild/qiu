"""Helpers to test sample-based phase propagators on the Aer simulator."""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer_encore.simulator import aer_simulator

from qiskit_pytest_helper.circuits import transpile_exactly

SIMULATOR_SEED = 1234
"""The seed of the simulated measurements, for reproducible tests."""


def run_propagator(
    propagator: QuantumCircuit, psi: Statevector
) -> tuple[bool, npt.NDArray[np.complex128]]:
    """Simulate a sample-based propagator on the initial state `psi`.

    The propagator acts on the registers `psi` and `phi` of `n` qubits each, with the
    `n` success flags as classical bits.

    Returns:
        Whether all cycles succeeded, i.e. the flags are 0, and the final amplitudes
        of the `psi` register, exact including their global phase, which requires the
        `phi` register to be back in `|0...0>`.
    """
    num_qubits = propagator.num_qubits // 2
    circuit = QuantumCircuit(*propagator.qregs, *propagator.cregs)
    circuit.initialize(psi, range(num_qubits))
    circuit.compose(propagator, circuit.qubits, circuit.clbits, inplace=True)
    circuit.save_statevector()  # type: ignore[attr-defined]

    simulator = aer_simulator(
        device="cpu", method="statevector", seed_simulator=SIMULATOR_SEED
    )
    result = simulator.run(transpile_exactly(circuit, simulator), shots=1).result()
    succeeded = result.get_counts().int_outcomes().get(0) == 1
    amplitudes = np.asarray(result.get_statevector().data)

    return succeeded, amplitudes[: 2**num_qubits]


def exact_cycles(
    psi: npt.ArrayLike, phi: npt.ArrayLike, deltas: npt.ArrayLike
) -> npt.NDArray[np.complex128]:
    """Return the normalized state after successful cycles of the protocol.

    Each successful cycle maps the amplitudes `psi_j` to
    `psi_j (1 + (e^(i delta) - 1) |phi_j|^2)`.
    """
    state: npt.NDArray[np.complex128] = np.asarray(psi, dtype=np.complex128)
    weights = np.abs(np.asarray(phi)) ** 2
    for delta in np.asarray(deltas, dtype=float):
        state = state * (1 + (np.exp(1j * delta) - 1) * weights)
        state = state / np.linalg.norm(state)
    return state
