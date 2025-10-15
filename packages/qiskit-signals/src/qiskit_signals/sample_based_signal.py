"""Module for sample-based quantum signals."""

from functools import cached_property

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import Statevector

from qiskit_signals.quantum_signal import GenericQuantumSignal


class ArbitrarySignalForSampleBasedProtocol(GenericQuantumSignal):
    """A generic quantum signal for sample-based protocol."""

    @classmethod
    def from_generic_signal(
        cls, signal: GenericQuantumSignal
    ) -> "ArbitrarySignalForSampleBasedProtocol":
        """Creates an ArbitrarySignalForSampleBasedProtocol from a GenericQuantumSignal."""
        return cls(axis=signal.axis, signal_function=signal.signal_function)

    @cached_property
    def alpha(self) -> float:
        """Returns the alpha coefficient of the quantum signal."""
        signal = self.data
        # gather all unique signs in the signal, either -1, 0, or 1
        signs = np.unique(np.sign(signal))
        # if 1 and -1 are both present, raise an error
        if 1 in signs and -1 in signs:
            raise ValueError(
                "All elements of the signal vector must have the same sign."
            )

        # normalization factor
        alpha = np.sum(signal)

        if np.isclose(alpha, 0):
            raise ValueError("The signal cannot be all zeros.")

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

    def __mul__(self, other: int | float) -> "ArbitrarySignalForSampleBasedProtocol":
        """Multiplies the quantum signal by a scalar."""
        if isinstance(other, (int, float)):

            def new_signal_function(x):
                return self.signal_function(x) * other

            new_signal = ArbitrarySignalForSampleBasedProtocol(
                self.axis, new_signal_function
            )

            return new_signal
        else:
            raise ValueError("Multiplication is only defined for scalars.")

    def __rmul__(self, other: int | float) -> "ArbitrarySignalForSampleBasedProtocol":
        """Right-hand multiplication so scalar * signal also works."""
        return self.__mul__(other)


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
