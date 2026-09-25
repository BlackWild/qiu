"""Direct phase circuits, applying `e^(i coef x^k)` for powers `k` up to 3.

The integer `x` encoded by a basis state is a weighted sum of its bits, see
`qubit_encoding`. Expanding `x^k` with `x_i^2 = x_i` for bits gives a sum of
products of at most `k` bits, each applied as a (multi-)controlled phase gate.
"""

from typing import ClassVar

from python_signals.algebraic_signal import PolynomialSignal
from python_signals.integer_axis import IndexOrdering
from qiskit.circuit import QuantumCircuit

from qiskit_phase_propagator.qubit_encoding import (
    bit_weights,
    is_msb_flipped,
    num_qubits_of,
)


class DirectPhase(QuantumCircuit):
    """Base class of the circuits applying `e^(i coef x^exponent)`.

    Subclasses define the `exponent` and apply the phases for the bit weights in
    `_apply_phases`.
    """

    exponent: ClassVar[int]
    """The power of the encoded integer in the phase.

    Not named `power`, which would shadow `QuantumCircuit.power`.
    """

    coef: float
    """The coefficient of the phase."""
    ordering: IndexOrdering
    """The index ordering encoding the integers in the basis states."""

    def __init__(self, num_qubits: int, coef: float, ordering: IndexOrdering) -> None:
        """Initialize the phase circuit.

        Args:
            num_qubits: The number of qubits encoding the integers.
            coef: The coefficient of the phase.
            ordering: The index ordering encoding the integers, see `qubit_encoding`.
        """
        super().__init__(num_qubits, name=f"direct_phase_{self.exponent}")
        self.coef = coef
        self.ordering = IndexOrdering(ordering)

        flip_msb = is_msb_flipped(self.ordering)
        if flip_msb:
            self.x(num_qubits - 1)
        self._apply_phases(coef, bit_weights(num_qubits, self.ordering))
        if flip_msb:
            self.x(num_qubits - 1)

    def _apply_phases(self, coef: float, weights: list[int]) -> None:
        raise NotImplementedError


class Order1DirectPhase(DirectPhase):
    """The phase circuit `e^(i coef x)`, with `x = sum_i w_i x_i`."""

    exponent = 1

    def _apply_phases(self, coef: float, weights: list[int]) -> None:
        for i, w_i in enumerate(weights):
            self.p(coef * w_i, i)


class Order2DirectPhase(DirectPhase):
    """The phase circuit `e^(i coef x^2)`.

    With `x^2 = sum_i w_i^2 x_i + 2 sum_(j<i) w_i w_j x_i x_j`.
    """

    exponent = 2

    def _apply_phases(self, coef: float, weights: list[int]) -> None:
        for i, w_i in enumerate(weights):
            self.p(coef * w_i**2, i)
            for j, w_j in enumerate(weights[:i]):
                self.cp(coef * 2 * w_i * w_j, i, j)


class Order3DirectPhase(DirectPhase):
    """The phase circuit `e^(i coef x^3)`.

    With `x^3 = sum_i w_i^3 x_i + 3 sum_(j<i) (w_i^2 w_j + w_i w_j^2) x_i x_j
    + 6 sum_(k<j<i) w_i w_j w_k x_i x_j x_k`.
    """

    exponent = 3

    def _apply_phases(self, coef: float, weights: list[int]) -> None:
        for i, w_i in enumerate(weights):
            self.p(coef * w_i**3, i)
            for j, w_j in enumerate(weights[:i]):
                self.cp(coef * 3 * (w_i**2 * w_j + w_i * w_j**2), i, j)
                for k, w_k in enumerate(weights[:j]):
                    self.mcp(coef * 6 * w_i * w_j * w_k, [i, j], k)


DIRECT_PHASES: dict[int, type[DirectPhase]] = {
    phase.exponent: phase
    for phase in (Order1DirectPhase, Order2DirectPhase, Order3DirectPhase)
}
"""The direct phase circuits by exponent."""


def polynomial_phase_circuit(signal: PolynomialSignal) -> QuantumCircuit:
    """Return the circuit applying `e^(i signal(x))` to the basis states of its axis.

    The phase of the basis state `|k>` is the signal at the axis value of the
    sample `k`, i.e. `e^(i alpha x_k^power)`.

    Args:
        signal: A monomial of power at most 3, on an axis of `2**n` samples.

    Returns:
        The phase circuit on `n` qubits.

    Raises:
        TypeError: If the signal is not a `PolynomialSignal`, e.g. a sum of monomials,
            whose circuits are composed instead.
        NotImplementedError: If its power is larger than 3.
    """
    if not isinstance(signal, PolynomialSignal):
        raise TypeError(
            "Direct phase circuits need a PolynomialSignal, i.e. a monomial, got "
            f"{type(signal).__name__}; compose the circuits of its monomials instead."
        )
    num_qubits = num_qubits_of(signal.axis)

    if signal.power == 0:
        circuit = QuantumCircuit(num_qubits, name="direct_phase_0")
        circuit.global_phase = signal.alpha
        return circuit

    phase = DIRECT_PHASES.get(signal.power)
    if phase is None:
        raise NotImplementedError(
            f"Direct phase circuits exist for powers up to 3, got {signal.power}."
        )
    return phase(num_qubits, signal.effective_alpha, signal.axis.ordering)
