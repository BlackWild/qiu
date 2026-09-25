"""The arithmetic of the sample-based phase protocol, independent of its implementation.

The protocol applies `e^(i f)` for a real signal `f` of one sign, decomposed as
`f = alpha |phi|**2` with the sum `alpha` of its samples and the normalized state
`phi = sqrt(f / alpha)`. Each cycle of a phase `delta`, post-selected on its success,
maps the amplitudes `psi_j` to `psi_j (1 + (e^(i delta) - 1) |phi_j|**2)`, renormalized,
which is `e^(i delta |phi_j|**2) psi_j` up to `O(delta**2)`. Slicing `alpha` into small
equal phases `delta` thus applies `e^(i alpha |phi|**2) = e^(i f)`.

See `qiskit_phase_propagator.sample_based` for its quantum circuits.
"""

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import SampledSignal


def decompose(signal: SampledSignal) -> tuple[float, npt.NDArray[np.float64]]:
    """Split a real signal of one sign into its sum and a normalized state.

    Args:
        signal: The signal `f`, real and either non-negative or non-positive.

    Returns:
        The sum `alpha` of the samples and the amplitudes `sqrt(f / alpha)`.

    Raises:
        ValueError: If the signal is not real, has samples of both signs, or vanishes.
    """
    data = np.asarray(signal.data)
    if np.iscomplexobj(data):
        if np.any(data.imag != 0):
            raise ValueError("The signal must be real.")
        data = data.real
    if np.any(data > 0) and np.any(data < 0):
        raise ValueError("The samples of the signal must all have the same sign.")
    alpha = float(np.sum(data))
    if alpha == 0:
        raise ValueError("The signal must not vanish.")
    return alpha, np.sqrt(data / alpha)


def slice_phase(alpha: float, max_delta: float) -> npt.NDArray[np.float64]:
    """Slice `alpha` into the fewest equal phases of magnitude at most `max_delta`."""
    if max_delta <= 0:
        raise ValueError(f"The max_delta must be positive, got {max_delta}.")
    number = int(np.ceil(abs(alpha) / max_delta))
    return np.full(number, alpha / number) if number else np.zeros(0)


def ideal_cycles(
    psi: npt.ArrayLike, phi: npt.ArrayLike, deltas: npt.ArrayLike
) -> tuple[npt.NDArray[np.complex128], float]:
    """Return the state after successful cycles, and their probability of success.

    Args:
        psi: The amplitudes of the state.
        phi: The amplitudes of the state `phi` of the decomposition.
        deltas: The phase of each cycle.

    Returns:
        The normalized state after the cycles, and the product of the probabilities of
        success of the cycles.
    """
    state = np.asarray(psi, dtype=np.complex128)
    weights = np.abs(np.asarray(phi)) ** 2
    probability = 1.0
    for delta in np.asarray(deltas, dtype=float):
        state = state * (1 + (np.exp(1j * delta) - 1) * weights)
        norm = float(np.linalg.norm(state))
        probability *= norm**2
        state = state / norm
    return state, probability
