"""The Qiskit backend of the lens experiment simulations.

The sample-based phase protocol is simulated on statevectors, post-selected on the
success of each cycle, with the circuits of `qiu_quantum_computing.phase_propagator`, see
`qiu_quantum_computing.phase_propagator.sample_based_manual`. The direct propagator is the circuit of
`qiu_hamiltonian_simulation`: an inverse QFT, a quadratic phase and a QFT.
"""

import numpy as np
from qiskit.quantum_info import Statevector
from qiu_classical_simulation.wave_optics.simulation import (
    PhaseOperation,
    PropagationBackend,
    State,
)
from qiu_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)
from qiu_qiskit_encore.synthesis_method import SynthesisMethod
from qiu_quantum_computing.phase_propagator.sample_based import (
    sample_based_decomposition,
    slice_alpha_to_deltas_evenly,
)
from qiu_quantum_computing.phase_propagator.sample_based_manual import (
    phase_propagation_cycle,
)
from qiu_quantum_computing.preparable_state import PreparableState
from qiu_signals.algebraic_signal import QuadraticSignal, SampledSignal


class QiskitBackend(PropagationBackend):
    """Applies the phases by statevector simulations of their quantum circuits."""

    state_preparation_method: SynthesisMethod
    """How the preparation of `|phi>` of the phase protocol is represented."""
    fourier_method: SynthesisMethod
    """How the Fourier transforms of the direct propagator are represented."""

    def __init__(
        self,
        state_preparation_method: SynthesisMethod = SynthesisMethod.DENSE,
        fourier_method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the backend.

        Args:
            state_preparation_method: How `|phi>` is prepared; the default `DENSE` is
                exact and fast on the few qubits of the transverse field.
            fourier_method: How the Fourier transforms of the direct propagator are
                represented.
        """
        self.state_preparation_method = state_preparation_method
        self.fourier_method = fourier_method

    def sample_based_phase(
        self, signal: SampledSignal, max_delta: float
    ) -> PhaseOperation:
        """Return the post-selected phase protocol applying `e^(i signal)`."""
        alpha, phi = sample_based_decomposition(signal)
        deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)
        preparable_phi = PreparableState(phi, self.state_preparation_method)

        def apply(state: State) -> tuple[State, float]:
            psi, probability = Statevector(state), 1.0
            for delta in deltas:
                psi, success = phase_propagation_cycle(psi, delta, preparable_phi)
                probability *= success
            return np.asarray(psi.data), probability

        return apply

    def direct_momentum_phase(self, signal: QuadraticSignal) -> PhaseOperation:
        """Return the evolution applying `e^(i signal)` to the angular spectrum."""
        evolution = MomentumDomainEvolutionQuadratic(signal, self.fourier_method)
        return lambda state: (
            np.asarray(Statevector(state).evolve(evolution).data),
            1.0,
        )
