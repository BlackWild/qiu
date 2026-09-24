"""Statevector simulation of the sample-based phase propagation, post-selected on success.

Instead of measuring the `phi` register, each cycle keeps the part of the state in
which it is `|0...0>`, i.e. the successful outcome, and renormalizes it. See
`sample_based` for the protocol.
"""

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import SampledSignal
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.synthesis_method import SynthesisMethod

from qiskit_phase_propagator.sample_based import (
    partial_phase_circuit,
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)


def phase_propagation_cycle(
    psi: Statevector, delta: float, phi: PreparableState
) -> tuple[Statevector, float]:
    """Simulate one cycle of the protocol, post-selected on its success.

    Args:
        psi: The state of the `psi` register.
        delta: The phase of the cycle.
        phi: The preparable state `|phi>`.

    Returns:
        The normalized state of the `psi` register after a successful cycle, and the
        probability of success.
    """
    num_qubits = phi.num_qubits
    phi_qubits = list(range(num_qubits, 2 * num_qubits))

    # the phi register holds the more significant qubits, prepared in |0...0>
    state = Statevector.from_label("0" * num_qubits).tensor(psi)
    state = state.evolve(phi.circuit, phi_qubits)
    state = state.evolve(partial_phase_circuit(delta, num_qubits))
    state = state.evolve(phi.inverse_circuit, phi_qubits)

    # the amplitudes with the phi register in |0...0> come first
    success = state.data[: 2**num_qubits]
    probability = float(np.vdot(success, success).real)
    return Statevector(success / np.sqrt(probability)), probability


def phase_propagate_state(
    psi_in: Statevector, deltas: npt.NDArray | list[float], phi: PreparableState
) -> Statevector:
    """Simulate one successful cycle per delta.

    Args:
        psi_in: The initial state of the `psi` register.
        deltas: The phase of each cycle.
        phi: The preparable state `|phi>`.

    Returns:
        The state of the `psi` register after all cycles.
    """
    psi = psi_in
    for delta in deltas:
        psi, _ = phase_propagation_cycle(psi, delta, phi)
    return psi


def phase_propagate_state_with_constant_delta(
    psi_in: Statevector, delta: float, num_cycles: int, phi: PreparableState
) -> Statevector:
    """Simulate `num_cycles` successful cycles with the same delta.

    Args:
        psi_in: The initial state of the `psi` register.
        delta: The phase of each cycle.
        num_cycles: The number of cycles.
        phi: The preparable state `|phi>`.

    Returns:
        The state of the `psi` register after all cycles.
    """
    return phase_propagate_state(psi_in, np.full(num_cycles, delta), phi)


def phase_propagate_state_with_arbitrary_signal(
    psi_in: Statevector,
    signal: SampledSignal,
    max_delta: float,
    method: SynthesisMethod = SynthesisMethod.GATE,
) -> Statevector:
    """Simulate the application of `e^(i f(x))` for a signal `f` of one sign.

    Args:
        psi_in: The initial state of the `psi` register.
        signal: The real signal `f` of one sign, on an axis of `2**n` samples.
        max_delta: The maximum phase per cycle.
        method: How the preparation of `|phi>` is represented in the circuits.

    Returns:
        The state of the `psi` register after all cycles.
    """
    alpha, state = sample_based_decomposition(signal)
    deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)
    return phase_propagate_state(psi_in, deltas, PreparableState(state, method))
