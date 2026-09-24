"""Quantum Fourier transform circuits.

Qiskit's quantum Fourier transform maps the basis state `|j>` of `n` qubits to
`sum_k e^(2 pi i j k / N) |k> / sqrt(N)` with `N = 2**n`, where the integers are
encoded in little-endian order like the indices of the statevector. On the
amplitudes, it is thus the orthonormal inverse discrete Fourier transform of NumPy,
`qft(psi) == numpy.fft.ifft(psi, norm="ortho")`, and the inverse QFT is
`numpy.fft.fft(psi, norm="ortho")`.
"""

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.synthesis.qft import synth_qft_full

from qiskit_encore.synthesis_method import SynthesisMethod


def qft_matrix(num_qubits: int) -> npt.NDArray[np.complex128]:
    """Return the dense unitary matrix of the quantum Fourier transform.

    Args:
        num_qubits: The number of qubits.

    Returns:
        The matrix with the entries `e^(2 pi i j k / N) / sqrt(N)`.
    """
    dimension = 2**num_qubits
    j = np.arange(dimension)
    return np.exp(2j * np.pi * np.outer(j, j) / dimension) / np.sqrt(dimension)


def qft_circuit(
    num_qubits: int,
    *,
    inverse: bool = False,
    method: SynthesisMethod = SynthesisMethod.GATE,
) -> QuantumCircuit:
    """Return a circuit of the quantum Fourier transform, see the module docstring.

    Args:
        num_qubits: The number of qubits, at least 1.
        inverse: If True, return the inverse quantum Fourier transform.
        method: How the transform is represented in the circuit:
            - `GATE` (default): a single Qiskit `QFTGate`, synthesized when
              transpiling.
            - `DECOMPOSED`: Hadamard, controlled phase and swap gates, synthesized
              by Qiskit's `synth_qft_full`.
            - `DENSE`: a single unitary gate of the matrix of `qft_matrix`.

    Returns:
        The quantum Fourier transform circuit.
    """
    if num_qubits < 1:
        raise ValueError(f"The number of qubits must be at least 1, got {num_qubits}.")
    method = SynthesisMethod(method)

    if method == SynthesisMethod.DECOMPOSED:
        return synth_qft_full(num_qubits, inverse=inverse)

    circuit = QuantumCircuit(num_qubits, name="iqft" if inverse else "qft")
    if method == SynthesisMethod.GATE:
        gate = QFTGate(num_qubits)
        circuit.append(gate.inverse() if inverse else gate, circuit.qubits)
    elif method == SynthesisMethod.DENSE:
        matrix = qft_matrix(num_qubits)
        circuit.unitary(
            matrix.conj().T if inverse else matrix,
            circuit.qubits,
            label="iqft" if inverse else "qft",
        )
    else:
        raise NotImplementedError(f"Unsupported synthesis method: {method.value!r}")
    return circuit
