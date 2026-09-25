"""Changes of basis between the position and the momentum domain.

For a position axis of `N = 2**n` samples, the momentum amplitudes are the discrete
Fourier transform `phi(p_k) = sum_j e^(-2 pi i j k / N) psi(x_j) / sqrt(N)`, i.e.
`numpy.fft.fft(psi, norm="ortho")`, with the momenta `p_k` in the `FFT` ordering of the
conjugate axis. This is Qiskit's *inverse* quantum Fourier transform, see
`qiskit_encore.qft`.
"""

from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PhysicalAxis
from qiskit.circuit import QuantumCircuit
from qiskit_encore.qft import qft_circuit
from qiskit_encore.synthesis_method import SynthesisMethod


def require_momentum_domain_axis(axis: PhysicalAxis) -> None:
    """Raise unless the axis is a Fourier conjugate axis in the `FFT` ordering.

    Only then does the basis state `|k>` after `to_momentum_basis` hold the sample
    `k` of a signal on the axis.

    Raises:
        ValueError: If the axis is not a Fourier domain axis, or if it is not in the
            `FFT` ordering.
    """
    if not axis.is_fourier_domain:
        raise ValueError(f"The axis must be a Fourier domain axis, got {axis!r}.")
    if axis.ordering != IndexOrdering.FFT:
        raise ValueError(
            f"The Fourier domain axis must be in the FFT ordering of the discrete "
            f"Fourier transform, got {axis!r}."
        )


def require_position_domain_axis(axis: PhysicalAxis) -> None:
    """Raise unless the axis is a position domain axis."""
    if axis.is_fourier_domain:
        raise ValueError(f"The axis must be a position domain axis, got {axis!r}.")


def to_momentum_basis(
    num_qubits: int, method: SynthesisMethod = SynthesisMethod.GATE
) -> QuantumCircuit:
    """Return the circuit mapping position amplitudes to momentum amplitudes."""
    return qft_circuit(num_qubits, inverse=True, method=method)


def to_position_basis(
    num_qubits: int, method: SynthesisMethod = SynthesisMethod.GATE
) -> QuantumCircuit:
    """Return the circuit mapping momentum amplitudes back to position amplitudes."""
    return qft_circuit(num_qubits, method=method)
