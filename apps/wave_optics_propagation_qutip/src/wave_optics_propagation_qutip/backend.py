"""The QuTiP backend of the lens experiment simulations.

The sample-based phase protocol acts on the ket `|phi> (x) |psi>` of an ancilla register
prepared in `|phi>` and the field: the partial phase multiplies the basis states where
both registers agree by `e^(i delta)`, and projecting the ancilla back onto `<phi|` keeps
the successful outcome of the cycle, see `python_wave_optics.phase_protocol`. The direct
propagator is the operator `F^dagger e^(i f) F` of the orthonormal DFT `F`.
"""

import numpy as np
import qutip as qt
from python_signals.algebraic_signal import QuadraticSignal, SampledSignal
from python_wave_optics.phase_protocol import decompose, slice_phase
from python_wave_optics.simulation import PhaseOperation, PropagationBackend, State


def partial_phase_operator(delta: float, dimension: int) -> qt.Qobj:
    """Return `e^(i delta)` on the states `|l> (x) |j>` with `l == j`, else 1."""
    diagonal = np.ones(dimension**2, dtype=np.complex128)
    diagonal[:: dimension + 1] = np.exp(1j * delta)
    return qt.qdiags(diagonal, 0, dims=[[dimension, dimension], [dimension, dimension]])


def dft_operator(dimension: int) -> qt.Qobj:
    """Return the orthonormal DFT, as `numpy.fft.fft(..., norm="ortho")` applies it."""
    return qt.Qobj(np.fft.fft(np.eye(dimension), axis=0, norm="ortho"))


class QutipBackend(PropagationBackend):
    """Applies the phases as QuTiP operators on kets."""

    def sample_based_phase(
        self, signal: SampledSignal, max_delta: float
    ) -> PhaseOperation:
        """Return the post-selected phase protocol applying `e^(i signal)`."""
        alpha, amplitudes = decompose(signal)
        deltas = slice_phase(alpha, max_delta)
        dimension = amplitudes.size
        phi = qt.Qobj(amplitudes.astype(np.complex128))
        # all deltas are equal, so one operator serves all cycles: the partial phase
        # followed by the projection of the ancilla onto <phi|
        cycle = qt.tensor(phi.dag(), qt.qeye(dimension)) @ partial_phase_operator(
            deltas[0] if deltas.size else 0.0, dimension
        )

        def apply(state: State) -> tuple[State, float]:
            psi, probability = qt.Qobj(state), 1.0
            for _ in deltas:
                successful = cycle @ qt.tensor(phi, psi)
                amplitudes_after = successful.full().reshape(-1)
                success = float(np.vdot(amplitudes_after, amplitudes_after).real)
                psi = qt.Qobj(amplitudes_after / np.sqrt(success))
                probability *= success
            return psi.full().reshape(-1), probability

        return apply

    def direct_momentum_phase(self, signal: QuadraticSignal) -> PhaseOperation:
        """Return the evolution applying `e^(i signal)` to the angular spectrum."""
        dft = dft_operator(signal.axis.size)
        evolution = dft.dag() @ qt.qdiags(np.exp(1j * signal.data), 0) @ dft
        return lambda state: ((evolution @ qt.Qobj(state)).full().reshape(-1), 1.0)
