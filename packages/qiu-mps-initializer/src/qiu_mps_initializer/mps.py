"""Matrix product states (MPS) of bond dimension 2 and the layers of gates preparing them.

A state of bond dimension at most 2 is prepared exactly by one layer of gates: a
two-qubit gate on each pair of neighboring qubits, from the most significant qubit
down, followed by a single-qubit gate on the least significant one (Ran, "Encoding of
matrix product states into quantum circuits of one- and two-qubit gates", Phys. Rev. A
101, 032310, 2020). Site `i` of an MPS is the qubit `n - 1 - i`, i.e. the MPS lists the
qubits from the most significant one, as the amplitudes of a Qiskit statevector do.
"""

import numpy as np
import numpy.typing as npt
import quimb.gates as gates
import quimb.tensor as qtn
import scipy.linalg
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector


def bond2_mps_approximation(
    state: Statevector | npt.ArrayLike,
) -> qtn.MatrixProductState:
    """Return the MPS of bond dimension at most 2 approximating a state of 2 qubits or more.

    The MPS is truncated from the state by successive singular value decompositions,
    normalized, right-canonical, and has its indices in the order left, physical, right.

    Args:
        state: The normalized state.

    Returns:
        The approximating MPS.

    Raises:
        ValueError: If the state is not a normalized state of 2 qubits or more.
    """
    statevector = validated_statevector(state)
    if statevector.num_qubits is None or statevector.num_qubits < 2:
        raise ValueError(
            "An MPS of bond dimension 2 needs a state of 2 qubits or more."
        )

    mps = qtn.MatrixProductState.from_dense(statevector.data, max_bond=2, absorb="left")
    mps.normalize()
    mps.right_canonicalize(inplace=True)
    mps.permute_arrays(shape="lpr")
    return mps


def disentangler_matrices(
    mps: qtn.MatrixProductState,
) -> list[npt.NDArray[np.complex128]]:
    """Return the unitaries of the layer preparing an MPS of bond dimension at most 2.

    The unitaries complete the tensors of the right-canonical MPS to unitary matrices:
    one `4 x 4` matrix per site but the last, acting on the qubits of the site and the
    next one, and a `2 x 2` matrix for the last site.

    Args:
        mps: A right-canonical MPS of 2 sites or more, of bond dimension at most 2, with
            its indices in the order left, physical, right, e.g. of
            `bond2_mps_approximation`. It is expanded to bond dimension 2 in place.

    Returns:
        The unitaries, from the first site to the last.

    Raises:
        ValueError: If the MPS has fewer than 2 sites, or if its bond dimension is
            larger than 2.
    """
    max_bond = mps.max_bond()
    if max_bond is None or mps.num_tensors < 2:
        raise ValueError("The MPS must have 2 sites or more.")
    if max_bond > 2:
        raise ValueError(
            f"The bond dimension of the MPS must be at most 2, got {max_bond}."
        )

    # bond dimensions of 1 are expanded, so that all tensors have the same shapes
    mps.expand_bond_dimension(2)

    # the first tensor, as the first column of a unitary, completed by its null space
    first = _site_data(mps, 0).reshape(4, 1)
    matrices = [np.column_stack((first, scipy.linalg.null_space(first.T).conjugate()))]

    # the middle tensors, as the first one or two columns, followed by a swap to act on
    # the qubit of the site and the next one
    for site in range(1, mps.num_tensors - 1):
        data = _site_data(mps, site)
        column_0, column_1 = data[0].reshape(4, 1), data[1].reshape(4, 1)
        columns = (
            column_0
            if np.allclose(column_1, 0)
            else np.column_stack((column_0, column_1))
        )
        completed = np.column_stack(
            (columns, scipy.linalg.null_space(columns.T).conjugate())
        )
        matrices.append(completed @ np.real(gates.SWAP))

    # the last tensor, of shape (2, 2); its second column vanishes if the bond was
    # expanded from dimension 1, and is then replaced by an orthogonal one
    last = _site_data(mps, mps.num_tensors - 1).T
    if np.allclose(last[:, 1], 0):
        orthogonal = scipy.linalg.null_space([last[:, 0].conjugate()])
        last = np.column_stack((last[:, 0], orthogonal))
    matrices.append(last)

    return [np.asarray(matrix, dtype=np.complex128) for matrix in matrices]


def _site_data(mps: qtn.MatrixProductState, site: int) -> npt.NDArray[np.complex128]:
    """Return the data of the tensor of a site."""
    return np.asarray(mps.tensors[site].data, dtype=np.complex128)


def mps_layer(mps: qtn.MatrixProductState) -> QuantumCircuit:
    """Return the layer of gates preparing an MPS of bond dimension at most 2.

    Args:
        mps: The MPS, see `disentangler_matrices`.

    Returns:
        The circuit `U` on one qubit per site with `U|0...0>` the state of the MPS.
    """
    matrices = disentangler_matrices(mps)
    num_qubits = len(matrices)
    circuit = QuantumCircuit(num_qubits, name="mps_layer")
    for site, matrix in enumerate(matrices[:-1]):
        qubit = num_qubits - 1 - site
        circuit.unitary(matrix, [qubit - 1, qubit], label=f"G{site}")
    circuit.unitary(matrices[-1], [0], label=f"G{num_qubits - 1}")
    return circuit
