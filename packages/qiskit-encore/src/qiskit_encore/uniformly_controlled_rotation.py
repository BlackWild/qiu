"""Uniformly controlled rotations, decomposed into elementary gates."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit

RotationAxis = Literal["y", "z"]
"""The axes a uniformly controlled rotation can rotate about."""


def uniformly_controlled_rotation(
    axis: RotationAxis, angles: npt.ArrayLike
) -> QuantumCircuit:
    """Return the uniformly controlled rotation with the given angles.

    The circuit acts on `k + 1` qubits for `2**k` angles: qubit 0 is the target, and
    qubits `1, ..., k` are the controls. For the controls in the basis state
    `|c>`, with `c = sum_j c_j 2^j` over the control qubits `j + 1`, the target is
    rotated by `R_axis(angles[c])`, i.e. the circuit implements the block diagonal
    unitary `diag(R(angles[0]), R(angles[1]), ...)`.

    The decomposition of Möttönen et al., "Quantum circuits for general multiqubit
    gates" (2004), uses `2**k` rotations and `2**k` CNOT gates for `k >= 1`, with
    the rotation angles computed by a Walsh-Hadamard transform. It involves no
    eigendecompositions, and is thus numerically robust for any angles.

    Args:
        axis: The rotation axis, `"y"` or `"z"`.
        angles: The `2**k` rotation angles, one per basis state of the controls.

    Returns:
        The circuit of `ry` or `rz` rotations and `cx` gates.

    Raises:
        ValueError: If the axis is not `"y"` or `"z"`, or if the angles are not a 1D
            array whose size is a power of 2.
    """
    if axis not in ("y", "z"):
        raise ValueError(f"The rotation axis must be 'y' or 'z', got {axis!r}.")

    angles = np.asarray(angles, dtype=float)
    num_controls = int(np.log2(angles.size)) if angles.size else -1
    if angles.ndim != 1 or angles.size != 2**num_controls:
        raise ValueError(
            f"The number of angles must be a power of 2, got shape {angles.shape}."
        )

    circuit = QuantumCircuit(num_controls + 1, name=f"UCR{axis.upper()}")
    rotate = circuit.ry if axis == "y" else circuit.rz

    # a rotation independent of the controls needs no CNOT gates
    if np.all(angles == angles[0]):
        if angles[0] != 0:
            rotate(angles[0], 0)
        return circuit

    # Along the Gray code g_0, g_1, ..., the CNOT gates flip the target for the
    # control bits set in g_i before the i-th rotation, which negates the angle.
    # The rotation for the control state c is thus sum_i (-1)^|c & g_i| theta_i,
    # i.e. theta solves the Walsh-Hadamard system M theta = angles.
    size = angles.size
    gray = [i ^ (i >> 1) for i in range(size)]
    walsh_hadamard = np.array(
        [[(-1) ** (c & g).bit_count() for g in gray] for c in range(size)]
    )
    thetas = walsh_hadamard.T @ angles / size

    for i, theta in enumerate(thetas):
        if theta != 0:
            rotate(theta, 0)
        if i + 1 < size:
            # g_i and g_(i+1) differ in the lowest set bit of i + 1
            changed_bit = ((i + 1) & -(i + 1)).bit_length() - 1
        else:
            # g_(size-1) = 2^(k-1) and g_0 = 0 differ in the highest bit
            changed_bit = num_controls - 1
        circuit.cx(changed_bit + 1, 0)

    return circuit
