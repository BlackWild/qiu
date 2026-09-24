"""The simulation of a lens experiment, independent of how the phases are applied.

The beam passes each lens slice, i.e. gets the slice's phase and propagates freely over
the slice thickness, and then propagates freely behind the lens in equal steps. The
phases are applied by a `PropagationBackend`, e.g. a quantum simulation of the
sample-based phase protocol, or `ExactBackend`, which applies them exactly.

Free propagation applies the phase of `free_space_propagator_phase` to the angular
spectrum, the orthonormal DFT of the field: directly if `direct_propagator` is set, and
otherwise with the sample-based phase protocol on the angular spectrum.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import QuadraticSignal, SampledSignal

from python_wave_optics.elements import free_space_propagator_phase
from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.result import (
    ExperimentResult,
    free_space_snapshot_name,
    lens_snapshot_name,
)

State = npt.NDArray[np.complex128]
"""The normalized amplitudes of the transverse field."""

PhaseOperation = Callable[[State], tuple[State, float]]
"""An operation on a state, returning the new state and its probability of success."""


class PropagationBackend(ABC):
    """Applies phase signals to states, e.g. by a quantum simulation."""

    @abstractmethod
    def sample_based_phase(
        self, signal: SampledSignal, max_delta: float
    ) -> PhaseOperation:
        """Return the operation applying `e^(i signal)` with the phase protocol.

        Args:
            signal: A real phase signal of one sign, on the axis of the state.
            max_delta: The maximum phase per cycle of the protocol.

        Returns:
            The operation, post-selected on the success of all cycles, returning the
            normalized state and the probability of success.
        """

    @abstractmethod
    def direct_momentum_phase(self, signal: QuadraticSignal) -> PhaseOperation:
        """Return the operation applying `e^(i signal)` to the angular spectrum.

        Args:
            signal: A quadratic phase signal on the angular wavenumber axis, in the FFT
                ordering of the orthonormal DFT of the state.

        Returns:
            The operation, with probability of success 1.
        """


def to_angular_spectrum(state: State) -> State:
    """Return the angular spectrum of a field, its orthonormal DFT."""
    return np.fft.fft(state, norm="ortho")


def from_angular_spectrum(spectrum: State) -> State:
    """Return the field of an angular spectrum, its orthonormal inverse DFT."""
    return np.fft.ifft(spectrum, norm="ortho")


class ExactBackend(PropagationBackend):
    """Applies the phases exactly, the classical reference of the simulations."""

    def sample_based_phase(
        self, signal: SampledSignal, max_delta: float
    ) -> PhaseOperation:
        """Return the exact multiplication by `e^(i signal)`, ignoring `max_delta`."""
        factors = np.exp(1j * np.asarray(signal.data))
        return lambda state: (state * factors, 1.0)

    def direct_momentum_phase(self, signal: QuadraticSignal) -> PhaseOperation:
        """Return the exact multiplication of the angular spectrum by `e^(i signal)`."""
        factors = np.exp(1j * np.asarray(signal.data))
        return lambda state: (
            from_angular_spectrum(factors * to_angular_spectrum(state)),
            1.0,
        )


def free_propagation(
    backend: PropagationBackend,
    parameters: ExperimentParameters,
    distance: float,
) -> PhaseOperation:
    """Return the free propagation over a distance in vacuum, as the parameters say."""
    signal = free_space_propagator_phase(
        parameters.k_axis, distance, parameters.vacuum_wavelength
    )
    if parameters.direct_propagator:
        return backend.direct_momentum_phase(signal)

    protocol = backend.sample_based_phase(signal, parameters.max_delta)

    def propagate(state: State) -> tuple[State, float]:
        spectrum, probability = protocol(to_angular_spectrum(state))
        return from_angular_spectrum(spectrum), probability

    return propagate


def has_phase(signal: SampledSignal) -> bool:
    """Whether a lens slice changes the field, i.e. its phase is not constant."""
    return not np.isclose(np.std(signal.data), 0)


def simulate(
    parameters: ExperimentParameters,
    backend: PropagationBackend,
    progress: Callable[[Iterable, str], Iterable] | None = None,
) -> ExperimentResult:
    """Simulate a lens experiment, taking a snapshot after each slice and step.

    Args:
        parameters: The experiment.
        backend: How the phases are applied.
        progress: Optionally wraps the loops over the lens slices and the free
            propagation steps with a description, e.g. to show a progress bar.

    Returns:
        The snapshots, and the cumulative success probabilities of the phase protocol.
    """
    track = progress or (lambda iterable, _: iterable)
    snapshots: dict[str, State] = {}
    probabilities: list[float] = []
    probability = 1.0

    def snapshot(name: str, state: State) -> None:
        snapshots[name] = state.copy()
        probabilities.append(probability)

    state = parameters.initial_state
    snapshot("step_0", state)

    total_lenses_simulated = 0
    slice_propagation = free_propagation(
        backend, parameters, parameters.lens_slice_thickness
    )
    for i, signal in enumerate(track(parameters.ordered_lens_signals, "Lens slices")):
        if has_phase(signal):
            state, success = backend.sample_based_phase(signal, parameters.max_delta)(
                state
            )
            probability *= success
            total_lenses_simulated += 1
        state, success = slice_propagation(state)
        probability *= success
        snapshot(lens_snapshot_name(i), state)
    snapshot("after_lens", state)

    step_propagation = free_propagation(
        backend, parameters, parameters.step_size_after_lens
    )
    for j in track(range(parameters.num_of_steps_after_lens), "Free space steps"):
        state, success = step_propagation(state)
        probability *= success
        snapshot(free_space_snapshot_name(j + 1), state)
    snapshot("final", state)

    return ExperimentResult(
        snapshots=snapshots,
        total_lenses_simulated=total_lenses_simulated,
        total_probability_of_success=probability,
        success_probabilities=probabilities,
    )
