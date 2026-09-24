"""Sample-based phase propagators, applying `e^(i f(x))` for a signal `f` of one sign.

The signal is split into its sum `alpha` and the state `|phi> = sqrt(f / alpha)` by
`sample_based_decomposition`, so that `f = alpha |phi|^2`. Each cycle of the protocol
prepares `|phi>` in a second register, applies the phase `e^(i delta)` where both
registers are in the same basis state (`partial_phase_circuit`), un-prepares `|phi>`
and measures the second register. On success, i.e. measuring `|0...0>`, the amplitudes
`psi_j` of the first register are mapped to `psi_j (1 + (e^(i delta) - 1) |phi_j|^2)`,
which is `e^(i delta |phi_j|^2) psi_j` up to `O(delta^2)`. Slicing `alpha` into small
`delta`s thus applies `e^(i alpha |phi|^2) = e^(i f)`.
"""

import numpy as np
import numpy.typing as npt
from python_signals.algebraic_signal import SampledSignal
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_encore.preparable_state import PreparableState
from qiskit_encore.synthesis_method import SynthesisMethod

from qiskit_phase_propagator.qubit_encoding import num_qubits_of


def sample_based_decomposition(signal: SampledSignal) -> tuple[float, Statevector]:
    """Split a real signal of one sign into its sum and a normalized state.

    Args:
        signal: The signal `f`, real and either non-negative or non-positive, on an
            axis of `2**n` samples.

    Returns:
        The sum `alpha` of the samples and the state `sqrt(f / alpha)`, such that
        `f = alpha |state|^2`.
    """
    num_qubits_of(signal.axis)
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

    return alpha, Statevector(np.sqrt(data / alpha))


def slice_alpha_to_deltas_evenly(alpha: float, max_delta: float) -> npt.NDArray:
    """Slice `alpha` into the fewest equal deltas of magnitude at most `max_delta`.

    Args:
        alpha: The total phase coefficient to slice.
        max_delta: The positive maximum magnitude of each delta.

    Returns:
        The equal deltas summing up to `alpha`, none if `alpha` is 0.
    """
    if max_delta <= 0:
        raise ValueError(f"The max_delta must be positive, got {max_delta}.")

    number_of_deltas = int(np.ceil(abs(alpha) / max_delta))
    if number_of_deltas == 0:
        return np.zeros(0)
    return np.full(number_of_deltas, alpha / number_of_deltas)


def partial_phase_diagonal(delta: float, num_qubits: int) -> npt.NDArray[np.complex128]:
    """Return the diagonal of the unitary of `partial_phase_circuit`.

    The entry of `|j>|l>`, at index `l * 2**n + j`, is `e^(i delta [j == l])`.

    Args:
        delta: The phase.
        num_qubits: The number of qubits `n` of each register.

    Returns:
        The `4**n` diagonal entries.
    """
    dimension = 2**num_qubits
    diagonal = np.ones(dimension**2, dtype=np.complex128)
    diagonal[:: dimension + 1] = np.exp(1j * delta)
    return diagonal


def partial_phase_circuit(delta: float, num_qubits: int) -> QuantumCircuit:
    """Return the phase `e^(i delta)` on the basis states where both registers agree.

    The circuit acts on the registers `psi` (qubits `0, ..., n-1`) and `phi` (qubits
    `n, ..., 2n-1`) and maps `|j>|l>` to `e^(i delta [j == l]) |j>|l>`.

    Args:
        delta: The phase.
        num_qubits: The number of qubits `n` of each register.

    Returns:
        The circuit on `2n` qubits.
    """
    psi_reg = QuantumRegister(num_qubits, name=r"\psi")
    phi_reg = QuantumRegister(num_qubits, name=r"\phi")
    circuit = QuantumCircuit(psi_reg, phi_reg, name="partial_phase")

    # flag the bits of psi that agree with phi, in place ...
    for psi_bit, phi_bit in zip(psi_reg, phi_reg, strict=True):
        circuit.cx(phi_bit, psi_bit, ctrl_state=0)
    # ... apply the phase if all of them agree ...
    if num_qubits == 1:
        circuit.p(delta, psi_reg[0])
    else:
        circuit.mcp(delta, psi_reg[:-1], psi_reg[-1])
    # ... and restore psi
    for psi_bit, phi_bit in zip(psi_reg, phi_reg, strict=True):
        circuit.cx(phi_bit, psi_bit, ctrl_state=0)

    return circuit


def _append_cycle(
    circuit: QuantumCircuit,
    delta: float,
    U_phi: QuantumCircuit,
    U_phi_dagger: QuantumCircuit,
    psi_reg: QuantumRegister,
    phi_reg: QuantumRegister,
) -> None:
    """Append the unitary part of one cycle: prepare, phase and un-prepare `|phi>`."""
    circuit.compose(U_phi, phi_reg, inplace=True)
    circuit.compose(
        partial_phase_circuit(delta, psi_reg.size), [*psi_reg, *phi_reg], inplace=True
    )
    circuit.compose(U_phi_dagger, phi_reg, inplace=True)


def propagator_registers(
    num_qubits: int,
) -> tuple[QuantumRegister, QuantumRegister, ClassicalRegister]:
    """Return the registers of a propagator: `psi`, `phi` and the success flags."""
    return (
        QuantumRegister(num_qubits, name=r"\psi"),
        QuantumRegister(num_qubits, name=r"\phi"),
        ClassicalRegister(num_qubits, name="success_flag"),
    )


class GenericIterativeSampleBasedPhasePropagator(QuantumCircuit):
    """A sample-based phase propagator with one cycle per delta.

    Each cycle only runs if all previous ones succeeded, i.e. measured the `phi`
    register in `|0...0>`. The `phi` register is reset after each cycle.
    """

    num_of_cycles: int
    """The number of cycles."""

    def __init__(
        self,
        deltas: npt.NDArray | list[float],
        U_phi: QuantumCircuit,
        U_phi_dagger: QuantumCircuit,
        take_snapshot: bool = False,
    ) -> None:
        """Initialize the propagator.

        Args:
            deltas: The phase of each cycle.
            U_phi: The circuit preparing `|phi>` from `|0...0>`.
            U_phi_dagger: The inverse of `U_phi`.
            take_snapshot: If True, save the statevector after each cycle, labeled
                by the cycle index (Aer simulators only).
        """
        psi_reg, phi_reg, success_flag = propagator_registers(U_phi.num_qubits)
        super().__init__(
            psi_reg, phi_reg, success_flag, name="Iterative phase propagator"
        )
        self.num_of_cycles = len(deltas)

        for cycle, delta in enumerate(deltas):
            with self.if_test((success_flag, 0)):
                _append_cycle(self, delta, U_phi, U_phi_dagger, psi_reg, phi_reg)
                self.measure(phi_reg, success_flag)
                # on success, phi is already |0...0>; the reset keeps the corrupted
                # output inspectable after a failure
                self.reset(phi_reg)
                if take_snapshot:
                    self.save_statevector(f"{cycle}")  # type: ignore[attr-defined]

    @classmethod
    def from_state(
        cls, state: PreparableState, deltas: npt.NDArray | list[float]
    ) -> "GenericIterativeSampleBasedPhasePropagator":
        """Create the propagator for the preparable state `|phi>`."""
        return cls(
            deltas=deltas, U_phi=state.circuit, U_phi_dagger=state.inverse_circuit
        )


class GenericIterativeSampleBasedPhasePropagatorWithConstantDelta(QuantumCircuit):
    """A sample-based phase propagator repeating one delta in a loop.

    The loop breaks at the first failed cycle, i.e. when the `phi` register is not
    measured in `|0...0>`.
    """

    num_of_cycles: int
    """The number of cycles."""

    def __init__(
        self,
        delta: float,
        number_of_cycles: int,
        U_phi: QuantumCircuit,
        U_phi_dagger: QuantumCircuit,
    ) -> None:
        """Initialize the propagator.

        Args:
            delta: The phase of each cycle.
            number_of_cycles: The number of cycles.
            U_phi: The circuit preparing `|phi>` from `|0...0>`.
            U_phi_dagger: The inverse of `U_phi`.
        """
        psi_reg, phi_reg, success_flag = propagator_registers(U_phi.num_qubits)
        super().__init__(
            psi_reg, phi_reg, success_flag, name="Iterative phase propagator"
        )
        self.num_of_cycles = number_of_cycles

        # Qiskit annotates the context manager form of for_loop too narrowly
        with self.for_loop(range(number_of_cycles)):  # pyright: ignore[reportCallIssue]
            _append_cycle(self, delta, U_phi, U_phi_dagger, psi_reg, phi_reg)
            self.measure(phi_reg, success_flag)
            self.reset(phi_reg)

            with self.if_test((success_flag, 0)) as else_:
                self.continue_loop()
            with else_:
                self.break_loop()

    @classmethod
    def from_state(
        cls, state: PreparableState, delta: float, number_of_cycles: int
    ) -> "GenericIterativeSampleBasedPhasePropagatorWithConstantDelta":
        """Create the propagator for the preparable state `|phi>`."""
        return cls(
            delta=delta,
            number_of_cycles=number_of_cycles,
            U_phi=state.circuit,
            U_phi_dagger=state.inverse_circuit,
        )


class QuadraticSignalSampleBasedPhasePropagator(QuantumCircuit):
    """The sample-based phase propagator applying `e^(i f(x))` for a signal `f`.

    The signal is decomposed as `f = alpha |phi|^2` by `sample_based_decomposition`,
    and `alpha` is sliced into equal deltas of magnitude at most `max_delta`.
    """

    num_of_cycles: int
    """The number of cycles."""

    def __init__(
        self,
        signal: SampledSignal,
        max_delta: float,
        method: SynthesisMethod = SynthesisMethod.GATE,
    ) -> None:
        """Initialize the propagator.

        Args:
            signal: The real signal `f` of one sign, on an axis of `2**n` samples.
            max_delta: The maximum phase per cycle.
            method: How the preparation of `|phi>` is represented in the circuit.
        """
        alpha, state = sample_based_decomposition(signal)
        deltas = slice_alpha_to_deltas_evenly(alpha, max_delta)

        psi_reg, phi_reg, success_flag = propagator_registers(
            num_qubits_of(signal.axis)
        )
        super().__init__(
            psi_reg, phi_reg, success_flag, name="Quadratic signal phase propagator"
        )

        propagator = (
            GenericIterativeSampleBasedPhasePropagatorWithConstantDelta.from_state(
                state=PreparableState(state, method=method),
                delta=deltas[0],
                number_of_cycles=len(deltas),
            )
        )
        self.num_of_cycles = propagator.num_of_cycles
        self.compose(propagator, inplace=True)
