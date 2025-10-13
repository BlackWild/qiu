"""Module for sample-based quantum signals."""

from functools import cached_property

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import Statevector

from qiskit_signals.quantum_signal import GenericQuantumSignal


class ArbitrarySignalForSampleBasedProtocol(GenericQuantumSignal):
    """A generic quantum signal for sample-based protocol."""

    def __init__(self, signal: GenericQuantumSignal) -> None:
        """Initializes the ArbitrarySignalForSampleBasedProtocol with the given signal data.

        Args:
            signal (GenericQuantumSignal): The generic quantum signal containing the signal data.
        """
        super().__init__(axis=signal.axis, signal_function=signal.signal_function)

    @cached_property
    def alpha(self) -> float:
        """Returns the alpha coefficient of the quantum signal."""
        signal = self.data
        # check if all elements of signal have the same sign
        if not all(
            [
                np.sign(signal[0]) == np.sign(signal[i]) or np.isclose(signal[i], 0)
                for i in range(len(signal))
            ]
        ):
            raise ValueError(
                "All elements of the signal vector must have the same sign."
            )

        # normalization factor
        alpha = np.sum(signal)

        return alpha

    @cached_property
    def statevector(self) -> Statevector:
        """Returns the statevector of the quantum signal."""
        signal = self.data

        # normalized signal
        normalized_signal = signal / self.alpha

        power = 2  # for sample-based protocol, we assume quadratic signal to get the corresponding wavefunction
        state_data = normalized_signal ** (1 / power)

        state = Statevector(state_data)
        return state


def extract_alpha_and_state_from_generic_signal(
    signal: npt.NDArray[np.float64], power: int
) -> tuple[float, Statevector]:
    """Extracts the alpha and the state from the generic signal."""

    # check if all elements of signal have the same sign
    if not all(
        [
            np.sign(signal[0]) == np.sign(signal[i]) or np.isclose(signal[i], 0)
            for i in range(len(signal))
        ]
    ):
        raise ValueError("All elements of the signal vector must have the same sign.")

    # normalization factor
    alpha = np.sum(signal)

    # normalized signal
    normalized_signal = signal / alpha

    # corresponding wavefunction
    state_data = normalized_signal ** (1 / power)

    state = Statevector(state_data)

    return alpha, state
